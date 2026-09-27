import unittest
from dataclasses import fields

import torch
import torch.nn.functional as F
from torch import nn

from spatiallm.model.chorus_fusion import ChorusFusionPointEncoder
from spatiallm.tuner.hparams.model_args import BaseModelArguments


def make_contrastive_encoder(
    *,
    batch_wide: bool = True,
    min_matched_tokens: int = 1,
    backprop_sonata: bool = False,
) -> ChorusFusionPointEncoder:
    encoder = ChorusFusionPointEncoder.__new__(ChorusFusionPointEncoder)
    nn.Module.__init__(encoder)
    encoder.contrastive_loss_weight = 1.0
    encoder.contrastive_exact_only = False
    encoder.contrastive_backprop_sonata = backprop_sonata
    encoder.contrastive_batch_wide_enabled = batch_wide
    encoder.contrastive_cosine_enabled = False
    encoder.contrastive_mse_enabled = False
    encoder.contrastive_info_nce_enabled = True
    encoder.contrastive_cosine_weight = 1.0
    encoder.contrastive_mse_weight = 1.0
    encoder.contrastive_info_nce_weight = 1.0
    encoder.contrastive_temperature = 0.5
    encoder.contrastive_min_matched_tokens = min_matched_tokens
    encoder.last_aux_loss = None
    encoder.last_aux_metrics = {}
    encoder.last_contrastive_pair = None
    return encoder


class BatchWideContrastiveLossTests(unittest.TestCase):
    def test_pooled_info_nce_matches_manual_symmetric_objective(self):
        encoder = make_contrastive_encoder()
        pred_a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        target_a = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
        pred_b = torch.tensor([[1.0, 1.0], [-1.0, 1.0]])
        target_b = torch.tensor([[0.8, 1.0], [-0.8, 1.0]])

        loss = encoder.finalize_batch_contrastive_loss(
            [(pred_a, target_a), (pred_b, target_b)]
        )

        pred = F.normalize(torch.cat([pred_a, pred_b]), p=2, dim=1)
        target = F.normalize(torch.cat([target_a, target_b]), p=2, dim=1)
        logits = pred @ target.T / encoder.contrastive_temperature
        labels = torch.arange(logits.shape[0])
        expected = 0.5 * (
            F.cross_entropy(logits, labels)
            + F.cross_entropy(logits.T, labels)
        )
        self.assertTrue(torch.allclose(loss, expected))
        self.assertEqual(encoder.last_aux_metrics["matched"].item(), 4.0)

    def test_other_scenes_supply_negatives(self):
        encoder = make_contrastive_encoder()
        pred_a = torch.tensor([[1.0, 0.0]])
        target_a = torch.tensor([[1.0, 0.0]])
        pred_b = torch.tensor([[1.0, 0.0]])
        target_b = torch.tensor([[1.0, 0.0]])

        scene_a_loss = encoder._compute_contrastive_loss_from_pairs(pred_a, target_a)
        scene_b_loss = encoder._compute_contrastive_loss_from_pairs(pred_b, target_b)
        pooled_loss = encoder.finalize_batch_contrastive_loss(
            [(pred_a, target_a), (pred_b, target_b)]
        )

        self.assertEqual(scene_a_loss.item(), 0.0)
        self.assertEqual(scene_b_loss.item(), 0.0)
        self.assertTrue(torch.allclose(pooled_loss, torch.log(torch.tensor(2.0))))

    def test_minimum_token_threshold_applies_after_pooling(self):
        encoder = make_contrastive_encoder(min_matched_tokens=4)
        pred_a = torch.eye(2)
        target_a = torch.eye(2)
        pred_b = -torch.eye(2)
        target_b = -torch.eye(2)

        self.assertIsNone(
            encoder._compute_contrastive_loss_from_pairs(pred_a, target_a)
        )
        self.assertIsNone(
            encoder._compute_contrastive_loss_from_pairs(pred_b, target_b)
        )
        pooled_loss = encoder.finalize_batch_contrastive_loss(
            [(pred_a, target_a), (pred_b, target_b)]
        )

        self.assertIsNotNone(pooled_loss)
        self.assertEqual(encoder.last_aux_metrics["matched"].item(), 4.0)

    def test_cosine_and_mse_are_averaged_over_all_pooled_pairs(self):
        encoder = make_contrastive_encoder()
        encoder.contrastive_cosine_enabled = True
        encoder.contrastive_mse_enabled = True
        encoder.contrastive_info_nce_enabled = False
        encoder.contrastive_cosine_weight = 0.75
        encoder.contrastive_mse_weight = 0.25
        pred_a = torch.tensor([[1.0, 0.0]])
        target_a = torch.tensor([[0.0, 1.0]])
        pred_b = torch.tensor(
            [[1.0, 1.0], [0.0, 1.0], [-1.0, 0.0]]
        )
        target_b = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
        )

        loss = encoder.finalize_batch_contrastive_loss(
            [(pred_a, target_a), (pred_b, target_b)]
        )

        pred = torch.cat([pred_a, pred_b])
        target = torch.cat([target_a, target_b])
        cosine = 1.0 - (
            F.normalize(pred, p=2, dim=1)
            * F.normalize(target, p=2, dim=1)
        ).sum(dim=1).mean()
        expected = 0.75 * cosine + 0.25 * F.mse_loss(pred, target)
        self.assertTrue(torch.allclose(loss, expected))

    def test_no_eligible_pairs_clears_auxiliary_loss(self):
        encoder = make_contrastive_encoder()
        encoder.last_aux_loss = torch.tensor(3.0)
        encoder.last_aux_metrics = {"matched": torch.tensor(10.0)}

        loss = encoder.finalize_batch_contrastive_loss([])

        self.assertIsNone(loss)
        self.assertIsNone(encoder.last_aux_loss)
        self.assertEqual(encoder.last_aux_metrics, {})

    def test_sonata_targets_are_detached_by_default(self):
        encoder = make_contrastive_encoder(backprop_sonata=False)
        chorus = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]],
            requires_grad=True,
        )
        sonata = torch.tensor(
            [[0.9, 0.1], [0.1, 0.9]],
            requires_grad=True,
        )
        indices = torch.arange(2)

        pair = encoder._prepare_contrastive_pair(
            chorus,
            sonata,
            indices,
            indices,
        )
        loss = encoder.finalize_batch_contrastive_loss([pair])
        loss.backward()

        self.assertIsNotNone(chorus.grad)
        self.assertIsNone(sonata.grad)

    def test_sonata_targets_receive_gradients_when_enabled(self):
        encoder = make_contrastive_encoder(backprop_sonata=True)
        chorus = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]],
            requires_grad=True,
        )
        sonata = torch.tensor(
            [[0.9, 0.1], [0.1, 0.9]],
            requires_grad=True,
        )
        indices = torch.arange(2)

        pair = encoder._prepare_contrastive_pair(
            chorus,
            sonata,
            indices,
            indices,
        )
        loss = encoder.finalize_batch_contrastive_loss([pair])
        loss.backward()

        self.assertIsNotNone(chorus.grad)
        self.assertIsNotNone(sonata.grad)

    def test_flag_off_keeps_scene_local_computation(self):
        encoder = make_contrastive_encoder(batch_wide=False)
        chorus = torch.eye(2)
        sonata = torch.eye(2)
        indices = torch.arange(2)

        loss = encoder._collect_or_compute_contrastive_loss(
            chorus,
            sonata,
            indices,
            indices,
        )

        self.assertIsNotNone(loss)
        self.assertIsNone(encoder.last_contrastive_pair)

    def test_batch_wide_mode_defers_each_scene_until_finalize(self):
        encoder = make_contrastive_encoder(min_matched_tokens=4)
        chorus = torch.eye(2)
        sonata = torch.eye(2)
        indices = torch.arange(2)

        loss = encoder._collect_or_compute_contrastive_loss(
            chorus,
            sonata,
            indices,
            indices,
        )

        self.assertIsNone(loss)
        self.assertIsNotNone(encoder.last_contrastive_pair)
        self.assertEqual(encoder.last_contrastive_pair[0].shape[0], 2)

    def test_model_argument_defaults_batch_wide_mode_to_false(self):
        field_by_name = {field.name: field for field in fields(BaseModelArguments)}

        self.assertIn(
            "chorus_contrastive_batch_wide_enabled",
            field_by_name,
        )
        self.assertFalse(
            field_by_name["chorus_contrastive_batch_wide_enabled"].default
        )


if __name__ == "__main__":
    unittest.main()
