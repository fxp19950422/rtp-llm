import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.triton_fused_executor import (
    _normalize_expert_routes,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_cp_router import (
    PureCpRouterInt8PerChannel,
    PureCpRouterNoQuant,
)


class PureCpRouterInt8PerChannelTest(unittest.TestCase):
    def test_remote_expert_slots_have_zero_weight(self):
        router = PureCpRouterInt8PerChannel.__new__(PureCpRouterInt8PerChannel)
        router.expert_start_id = 4
        router.expert_num_per_rank = 2

        hidden_states = torch.ones((2, 4), dtype=torch.bfloat16)
        topk_ids = torch.tensor([[4, 7], [2, 5]], dtype=torch.int32)
        topk_weights = torch.tensor([[0.7, 0.3], [0.4, 0.6]], dtype=torch.float32)
        adjusted_ids = torch.tensor([[0, -1], [-1, 1]], dtype=torch.int32)
        expert_counts = torch.tensor([1, 1], dtype=torch.int32)

        with (
            patch(
                "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_cp_router.all_gather",
                side_effect=lambda value, group: value,
            ),
            patch(
                "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_cp_router.recompute_topk_ids_sum_expert_count",
                return_value=(adjusted_ids, expert_counts),
            ),
        ):
            payload = router.prepare(hidden_states, None, None, topk_weights, topk_ids)

        self.assertTrue(payload.expert_ids_are_local)
        self.assertTrue(torch.equal(payload.expert_topk_ids, adjusted_ids))
        torch.testing.assert_close(
            payload.expert_topk_weights,
            torch.tensor([[0.7, 0.0], [0.0, 0.6]]),
        )

    def test_no_quant_router_marks_local_expert_ids(self):
        router = PureCpRouterNoQuant.__new__(PureCpRouterNoQuant)
        router.expert_start_id = 4
        router.expert_num_per_rank = 2

        hidden_states = torch.ones((1, 4), dtype=torch.bfloat16)
        topk_ids = torch.tensor([[4, 7]], dtype=torch.int32)
        topk_weights = torch.tensor([[0.7, 0.3]], dtype=torch.float32)
        adjusted_ids = torch.tensor([[0, -1]], dtype=torch.int32)
        expert_counts = torch.tensor([1, 0], dtype=torch.int32)

        with (
            patch(
                "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_cp_router.all_gather",
                side_effect=lambda value, group: value,
            ),
            patch(
                "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_cp_router.recompute_topk_ids_sum_expert_count",
                return_value=(adjusted_ids, expert_counts),
            ),
        ):
            payload = router.prepare(hidden_states, None, None, topk_weights, topk_ids)

        self.assertTrue(payload.expert_ids_are_local)
        self.assertIs(payload.expert_x, hidden_states)
        self.assertTrue(torch.equal(payload.expert_topk_ids, adjusted_ids))
        torch.testing.assert_close(
            payload.expert_topk_weights, torch.tensor([[0.7, 0.0]])
        )

    def test_triton_executor_preserves_prelocalized_ids(self):
        topk_ids = torch.tensor([[0, -1], [1, -1]], dtype=torch.int32)
        topk_weights = torch.tensor([[0.7, 0.3], [0.6, 0.4]], dtype=torch.float32)

        local_ids, local_weights = _normalize_expert_routes(
            topk_ids,
            topk_weights,
            start_expert_id=4,
            num_experts_per_partition=2,
            expert_ids_are_local=True,
        )

        self.assertTrue(
            torch.equal(local_ids, torch.tensor([[0, 0], [1, 0]], dtype=torch.int32))
        )
        torch.testing.assert_close(
            local_weights,
            torch.tensor([[0.7, 0.0], [0.6, 0.0]], dtype=torch.float32),
        )


if __name__ == "__main__":
    unittest.main()
