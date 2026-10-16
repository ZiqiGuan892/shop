"""Check full SFT configuration and export with fake dependencies, never train a model."""

import builtins
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from scripts import train_sft
from shopping_grpo.evaluation.blind_guard import validate_canonical_blind_asset


class Parameter:
    def __init__(self, size, requires_grad=False):
        self.size, self.requires_grad = size, requires_grad

    def requires_grad_(self, enabled):
        self.requires_grad = enabled

    def numel(self):
        return self.size


class Model:
    def __init__(self):
        self.weights = [Parameter(10), Parameter(20)]
        self.config = SimpleNamespace(use_cache=True)

    def parameters(self):
        return iter(self.weights)


class FullSftTests(unittest.TestCase):
    def args(self, *extra):
        return train_sft.parse_args([
            "--model", "test-model", "--train", "train.jsonl", "--output", "output", *extra,
        ])

    def test_full_is_default_has_separate_lr_and_rejects_qlora(self):
        args = self.args()
        self.assertEqual(args.finetuning_type, "full")
        self.assertEqual(args.learning_rate, 2e-5)
        self.assertEqual(self.args("--finetuning-type", "lora").learning_rate, 1e-4)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.args("--qlora")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.args("--finetuning-t", "lora")

    def test_full_unfreezes_original_parameters_without_peft_or_adapter(self):
        model = Model()
        forbidden = MagicMock(side_effect=AssertionError("PEFT must not be called"))
        with redirect_stdout(io.StringIO()):
            configured, counts = train_sft._configure_trainable_model(
                model, self.args(), lora_config=forbidden, get_peft_model=forbidden,
            )
        self.assertIs(configured, model)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertEqual(counts, {"total": 30, "trainable": 30})
        forbidden.assert_not_called()

    def test_full_multimodal_ddp_handles_unused_visual_parameters(self):
        args = self.args("--gradient-checkpointing")
        kwargs = train_sft._training_argument_kwargs(args, "bf16", True, True)
        self.assertTrue(kwargs["ddp_find_unused_parameters"])
        self.assertEqual(kwargs["gradient_checkpointing_kwargs"], {"use_reentrant": False})
        self.assertTrue(kwargs["prediction_loss_only"])
        self.assertEqual(kwargs["optim"], "adamw_torch")
        self.assertEqual(kwargs["eval_strategy"], "epoch")
        self.assertTrue(kwargs["bf16"])
        self.assertNotIn("deepspeed", kwargs)

    def test_npu_is_bound_before_dtype_detection_and_full_never_imports_peft(self):
        calls = []

        def bf16_supported():
            self.assertEqual(calls, [("set_device", 3)])
            return True

        torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            npu=SimpleNamespace(is_available=lambda: True,
                                set_device=lambda rank: calls.append(("set_device", rank)),
                                is_bf16_supported=bf16_supported),
            bfloat16="bf16", float16="fp16", float32="fp32",
        )
        transformers = SimpleNamespace(**{name: object() for name in (
            "AutoConfig", "AutoModelForCausalLM", "AutoModelForMultimodalLM", "AutoProcessor",
            "AutoTokenizer", "BitsAndBytesConfig", "Trainer", "TrainerCallback", "TrainingArguments",
        )})
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name == "peft":
                raise AssertionError("full training imported PEFT")
            return original_import(name, *args, **kwargs)

        with patch.dict(sys.modules, {"torch": torch, "transformers": transformers}), \
                patch("scripts.train_sft.find_spec", return_value=None), \
                patch.dict(os.environ, {"LOCAL_RANK": "3"}), \
                patch("builtins.__import__", side_effect=guarded_import):
            deps = train_sft._training_dependencies("full")
            self.assertTrue(all(value is None for value in deps[1:5]))
            self.assertEqual(train_sft._resolve_dtype(self.args(), torch), ("bf16", "bf16"))

    def test_data_splits_reject_overlap_and_final_evaluation_tasks(self):
        _, reserved = validate_canonical_blind_asset()
        safe = next(x for x in range(1000) if x not in reserved)
        final = next(iter(reserved))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train, validation = root / "train.jsonl", root / "validation.jsonl"
            train.write_text(json.dumps({"task_id": safe}) + "\n", encoding="utf-8")
            validation.write_text(json.dumps({"task_id": safe}) + "\n", encoding="utf-8")
            args = SimpleNamespace(train=train, validation=validation)
            with self.assertRaisesRegex(SystemExit, "重叠"):
                train_sft._validate_sft_splits(args)
            validation.write_text(json.dumps({"task_id": final}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "评测任务"):
                train_sft._validate_sft_splits(args)

    def test_zero3_requires_complete_weight_gather_for_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ds.json"
            path.write_text(json.dumps({"zero_optimization": {"stage": 3}}), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "stage3_gather"):
                train_sft._validate_deepspeed_config(path)
            path.write_text(json.dumps({"zero_optimization": {
                "stage": 3, "stage3_gather_16bit_weights_on_model_save": True,
            }}), encoding="utf-8")
            train_sft._validate_deepspeed_config(path)
            args = self.args("--deepspeed", str(path))
            self.assertEqual(train_sft._training_argument_kwargs(args, "bf16", True, True)["deepspeed"], str(path))

    def test_trainer_exports_full_model_and_only_rank_zero_writes_processor_and_summary(self):
        for rank in (0, 1):
            with self.subTest(rank=rank), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                output = root / "model"
                model = Model()
                processor = MagicMock()
                tokenizer = SimpleNamespace(pad_token_id=0, eos_token="eos", padding_side="right")
                torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False),
                                        bfloat16="bf16", float16="fp16", float32="fp32")
                order = []

                def training_arguments(**kwargs):
                    order.append("training_args")
                    return SimpleNamespace(**kwargs, process_index=rank)

                def load_model(*args, **kwargs):
                    self.assertEqual(order, ["training_args"])
                    order.append("model")
                    return model

                model_class = SimpleNamespace(from_pretrained=load_model)
                trainer = MagicMock()
                trainer.train.return_value = SimpleNamespace(training_loss=0.1, metrics={})
                trainer.state.log_history = []
                trainer.is_world_process_zero.return_value = rank == 0
                trainer_class = MagicMock(return_value=trainer)
                deps = (torch, None, None, None, None, None, model_class, model_class,
                        None, None, None, object, object, training_arguments)
                with patch.object(sys, "argv", ["train_sft.py", "--model", "test-model",
                                               "--train", "unused", "--output", str(output)]), \
                        patch("scripts.train_sft._validate_sft_splits"), \
                        patch("scripts.train_sft.guard_blind_final"), \
                        patch("scripts.train_sft._training_dependencies", return_value=deps) as dependency_loader, \
                        patch("scripts.train_sft._load_preprocessing_components", return_value=(tokenizer, processor, True)), \
                        patch("scripts.train_sft.load_supervised_examples", return_value=([{"task_id": 1}], {})), \
                        patch("scripts.train_sft._torch_dataset", side_effect=lambda examples, torch: examples), \
                        patch("scripts.train_sft._loss_only_eval_trainer_class", return_value=trainer_class), \
                        redirect_stdout(io.StringIO()):
                    train_sft.main()
                dependency_loader.assert_called_once_with("full")
                self.assertIs(trainer_class.call_args.kwargs["model"], model)
                self.assertTrue(all(p.requires_grad for p in model.parameters()))
                trainer.save_model.assert_called_once_with(str(output))
                if rank == 0:
                    processor.save_pretrained.assert_called_once_with(str(output))
                    summary = json.loads((output / "train_summary.json").read_text(encoding="utf-8"))
                    self.assertEqual(summary["finetuning_type"], "full")
                    self.assertEqual(summary["parameter_counts"], {"total": 30, "trainable": 30})
                else:
                    processor.save_pretrained.assert_not_called()
                    self.assertFalse((output / "train_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
