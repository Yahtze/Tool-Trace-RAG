from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from tool_trace_rag.config import CUSTOMER_SUPPORT_DATA_PATH
from tool_trace_rag.tools.registry import ToolDefinition, ToolRegistry


class CustomerSupportAdapter(Protocol):
    def customers(self) -> list[dict[str, Any]]: ...

    def orders(self) -> list[dict[str, Any]]: ...

    def policies(self) -> dict[str, Any]: ...


class JsonCustomerSupportAdapter:
    def __init__(self, data_path: str | Path) -> None:
        self._data = json.loads(Path(data_path).read_text(encoding="utf-8"))

    def customers(self) -> list[dict[str, Any]]:
        return self._data["customers"]

    def orders(self) -> list[dict[str, Any]]:
        return self._data["orders"]

    def policies(self) -> dict[str, Any]:
        return self._data["policies"]


class InMemoryCustomerSupportAdapter:
    def __init__(
        self,
        *,
        customers: list[dict[str, Any]],
        orders: list[dict[str, Any]],
        policies: dict[str, Any],
    ) -> None:
        self._customers = customers
        self._orders = orders
        self._policies = policies

    def customers(self) -> list[dict[str, Any]]:
        return self._customers

    def orders(self) -> list[dict[str, Any]]:
        return self._orders

    def policies(self) -> dict[str, Any]:
        return self._policies


class RefundPolicyModule:
    def __init__(self, adapter: CustomerSupportAdapter) -> None:
        self._adapter = adapter

    def find_customer(self, query: str) -> dict[str, Any]:
        normalized_query = query.strip().lower()
        for customer in self._adapter.customers():
            if normalized_query in {
                customer["customer_id"].lower(),
                customer["name"].lower(),
                customer["email"].lower(),
            }:
                return {"status": "found", "customer": customer}

        return {
            "status": "not_found",
            "error_code": "CUSTOMER_NOT_FOUND",
            "message": "No customer matched query.",
        }

    def get_customer_orders(self, customer_id: str) -> dict[str, Any]:
        customer_orders = [order for order in self._adapter.orders() if order["customer_id"] == customer_id]
        if not customer_orders:
            return {
                "status": "not_found",
                "error_code": "CUSTOMER_ORDERS_NOT_FOUND",
                "message": "No orders matched customer_id.",
            }

        summaries = [
            {
                "order_id": order["order_id"],
                "item": order["item"],
                "category": order["category"],
                "status": order["status"],
                "delivered_days_ago": order["delivered_days_ago"],
            }
            for order in customer_orders
        ]
        return {"status": "found", "customer_id": customer_id, "orders": summaries}

    def get_order(self, order_id: str) -> dict[str, Any]:
        order = self._find_order(order_id)
        if order is None:
            return {
                "status": "not_found",
                "error_code": "ORDER_NOT_FOUND",
                "message": "No order matched order_id.",
            }
        return {"status": "found", "order": order}

    def check_refund_eligibility(self, order_id: str) -> dict[str, Any]:
        order = self._find_order(order_id)
        if order is None:
            return {
                "status": "not_found",
                "error_code": "ORDER_NOT_FOUND",
                "message": "No order matched order_id.",
            }

        policies = self._adapter.policies()
        if order["status"] in policies["not_delivered_statuses"]:
            return self._refund_result(order_id, False, "Order has not been delivered yet.")
        if order["status"] == "cancelled":
            return self._refund_result(order_id, False, "Cancelled orders are not refundable.")
        if order["category"] in policies["final_sale_categories"]:
            return self._refund_result(order_id, False, "Final sale items are not refundable.")
        if order["category"] == "electronics" and order["opened"] and not policies["opened_electronics_refundable"]:
            return self._refund_result(order_id, False, "Opened electronics are not refundable.")
        if order["delivered_days_ago"] is None:
            return self._refund_result(order_id, False, "Order has not been delivered yet.")
        if order["delivered_days_ago"] > policies["standard_refund_window_days"]:
            return self._refund_result(order_id, False, "Order is outside the 30-day refund window.")
        return self._refund_result(order_id, True, "Order is within the 30-day refund window.")

    def _find_order(self, order_id: str) -> dict[str, Any] | None:
        return next((order for order in self._adapter.orders() if order["order_id"] == order_id), None)

    @staticmethod
    def _refund_result(order_id: str, eligible: bool, reason: str) -> dict[str, Any]:
        return {"status": "ok", "order_id": order_id, "eligible": eligible, "reason": reason}


def build_customer_support_registry(data_path: str | Path = CUSTOMER_SUPPORT_DATA_PATH) -> ToolRegistry:
    return build_customer_support_registry_from_adapter(JsonCustomerSupportAdapter(data_path))


def build_customer_support_registry_from_adapter(adapter: CustomerSupportAdapter) -> ToolRegistry:
    module = RefundPolicyModule(adapter)
    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            name="find_customer",
            description="Find a customer by customer ID, exact name, or email address.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Customer ID, exact name, or email."}},
                "required": ["query"],
                "additionalProperties": False,
            },
            function=module.find_customer,
        )
    )
    registry.register(
        ToolDefinition(
            name="get_customer_orders",
            description="Return compact order summaries for a customer.",
            parameters={
                "type": "object",
                "properties": {"customer_id": {"type": "string"}},
                "required": ["customer_id"],
                "additionalProperties": False,
            },
            function=module.get_customer_orders,
        )
    )
    registry.register(
        ToolDefinition(
            name="get_order",
            description="Return the full record for an order.",
            parameters={
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
                "additionalProperties": False,
            },
            function=module.get_order,
        )
    )
    registry.register(
        ToolDefinition(
            name="check_refund_eligibility",
            description="Check whether an order is eligible for refund under the store policy.",
            parameters={
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
                "additionalProperties": False,
            },
            function=module.check_refund_eligibility,
        )
    )
    return registry
