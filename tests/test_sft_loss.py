"""Numerical and saved-activation checks for SFT loss, using small CPU tensors."""

import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from torch.nn import functional as F

    from shopping_grpo.training.sft.loss import chunked_causal_lm_loss


def reference_causal_lm_loss(
    logits, labels, vocab_size, num_items_in_batch=None, ignore_index=-100,
    shift_labels=None,
):
    """HF ForCausalLMLoss's unchunked FP32 cross entropy and normalization."""
    if shift_labels is None:
        shift_labels = F.pad(labels, (0, 1), value=ignore_index)[..., 1:]
    loss = F.cross_entropy(
        logits.float().reshape(-1, vocab_size),
        shift_labels.reshape(-1).to(logits.device),
        ignore_index=ignore_index,
        reduction="mean" if num_items_in_batch is None else "sum",
    )
    if num_items_in_batch is not None:
        loss = loss / num_items_in_batch
    return loss


@unittest.skipIf(torch is None, "optional PyTorch dependency is unavailable")
class ChunkedSftLossTest(unittest.TestCase):
    def assert_loss_and_gradient_match(self, values, labels, *, chunk_size=4, **kwargs):
        reference_logits = values.detach().clone().requires_grad_()
        chunked_logits = values.detach().clone().requires_grad_()
        expected = reference_causal_lm_loss(
            reference_logits, labels, values.shape[-1], **kwargs,
        )
        actual = chunked_causal_lm_loss(
            chunked_logits, labels, values.shape[-1], chunk_size=chunk_size, **kwargs,
        )
        expected.backward()
        actual.backward()
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6, equal_nan=True)
        # BF16 casts can round differently after summing token chunks.
        tolerance = 1e-2 if values.dtype == torch.bfloat16 else 1e-6
        torch.testing.assert_close(
            chunked_logits.grad, reference_logits.grad,
            atol=tolerance * 1e-2, rtol=tolerance,
        )
        return actual, chunked_logits.grad

    def test_float32_and_bfloat16_match_with_multiple_examples_and_uneven_chunks(self):
        generator = torch.Generator().manual_seed(17)
        values = torch.randn(2, 7, 19, generator=generator)
        labels = torch.tensor([
            [-100, 3, 8, -100, 11, 2, -100],
            [-100, -100, 5, 0, -100, 18, 7],
        ])
        for dtype in (torch.float32, torch.bfloat16):
            for chunk_size in (1, 4, 128):
                with self.subTest(dtype=dtype, chunk_size=chunk_size):
                    self.assert_loss_and_gradient_match(
                        values.to(dtype), labels, chunk_size=chunk_size,
                    )

    def test_all_ignored_chunk_does_not_change_global_normalization(self):
        values = torch.randn(1, 11, 13, generator=torch.Generator().manual_seed(2))
        labels = torch.tensor([[-100, -100, -100, -100, -100, 3, 5, 9, -100, 1, 7]])
        self.assert_loss_and_gradient_match(values, labels, chunk_size=4)

    def test_external_denominator_matches_sum_over_all_chunks(self):
        values = torch.randn(2, 5, 13, generator=torch.Generator().manual_seed(3))
        labels = torch.tensor([[-100, 2, 5, -100, 7], [-100, 8, 1, 10, -100]])
        for denominator in (23, torch.tensor(23), torch.tensor(23.0)):
            with self.subTest(denominator=denominator):
                self.assert_loss_and_gradient_match(
                    values, labels, num_items_in_batch=denominator,
                )

    def test_explicit_shift_labels_are_not_shifted_again(self):
        values = torch.randn(2, 5, 11, generator=torch.Generator().manual_seed(4))
        labels = torch.full((2, 5), -100)
        shifted = torch.tensor([[3, 0, -100, 1, -100], [-100, 8, 2, -100, 10]])
        self.assert_loss_and_gradient_match(values, labels, shift_labels=shifted)

    def test_custom_ignore_index(self):
        values = torch.randn(2, 4, 11, generator=torch.Generator().manual_seed(5))
        labels = torch.tensor([[-7, 2, -7, 5], [-7, -7, 8, 1]])
        self.assert_loss_and_gradient_match(values, labels, ignore_index=-7)

    def test_noncontiguous_logits(self):
        values = torch.randn(2, 11, 7, generator=torch.Generator().manual_seed(6)).transpose(1, 2)
        self.assertFalse(values.is_contiguous())
        labels = torch.tensor([[-100, 1, 3, 8, -100, 2, 10], [-100, 4, 0, -100, 3, 9, 7]])
        self.assert_loss_and_gradient_match(values, labels)

    def test_evaluation_without_gradients_matches(self):
        values = torch.randn(2, 7, 11, generator=torch.Generator().manual_seed(7), requires_grad=True)
        labels = torch.tensor([[-100, 1, 3, 8, -100, 2, 10], [-100, 4, 0, -100, 3, 9, 7]])
        with torch.no_grad():
            expected = reference_causal_lm_loss(values, labels, 11)
            actual = chunked_causal_lm_loss(values, labels, 11, chunk_size=3)
        self.assertFalse(actual.requires_grad)
        torch.testing.assert_close(actual, expected)

    def test_logits_without_requires_grad_match(self):
        values = torch.randn(1, 7, 11, generator=torch.Generator().manual_seed(8))
        labels = torch.tensor([[-100, 1, 3, 8, -100, 2, 10]])
        actual = chunked_causal_lm_loss(values, labels, 11, chunk_size=3)
        self.assertFalse(actual.requires_grad)
        torch.testing.assert_close(actual, reference_causal_lm_loss(values, labels, 11))

    def test_all_ignored_mean_matches_nan_loss_and_zero_gradients(self):
        values = torch.randn(2, 7, 11, generator=torch.Generator().manual_seed(9))
        labels = torch.full((2, 7), -100)
        actual, gradient = self.assert_loss_and_gradient_match(values, labels)
        self.assertTrue(torch.isnan(actual))
        self.assertTrue(torch.equal(gradient, torch.zeros_like(gradient)))

    def test_all_ignored_external_denominator_is_zero(self):
        values = torch.randn(2, 7, 11, generator=torch.Generator().manual_seed(10))
        labels = torch.full((2, 7), -100)
        actual, gradient = self.assert_loss_and_gradient_match(
            values, labels, num_items_in_batch=17,
        )
        self.assertEqual(actual.item(), 0.0)
        self.assertTrue(torch.equal(gradient, torch.zeros_like(gradient)))

    def test_checkpointing_bounds_saved_fp32_cross_entropy_activations(self):
        """A plain chunk loop would still retain every chunk's FP32 softmax."""
        sequence_length, vocabulary, chunk_size = 137, 67, 16
        values = torch.randn(
            1, sequence_length, vocabulary, generator=torch.Generator().manual_seed(11),
            dtype=torch.bfloat16,
        )
        labels = torch.arange(sequence_length).reshape(1, -1) % vocabulary

        def saved_fp32_elements(loss_function):
            saved = []

            def pack(tensor):
                saved.append((tensor.dtype, tensor.numel()))
                return tensor

            with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                loss = loss_function(values.detach().clone().requires_grad_())
            loss.backward()
            return sum(size for dtype, size in saved if dtype == torch.float32)

        reference_saved = saved_fp32_elements(
            lambda logits: reference_causal_lm_loss(logits, labels, vocabulary)
        )
        chunked_saved = saved_fp32_elements(
            lambda logits: chunked_causal_lm_loss(
                logits, labels, vocabulary, chunk_size=chunk_size,
            )
        )
        self.assertGreaterEqual(reference_saved, sequence_length * vocabulary)
        self.assertLess(chunked_saved, 2 * chunk_size * vocabulary)
        self.assertLess(chunked_saved, reference_saved)


if __name__ == "__main__":
    unittest.main()
