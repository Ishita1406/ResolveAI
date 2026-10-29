from enum import StrEnum


class OrderStatus(StrEnum):
    PLACED = "placed"
    SHIPPED = "shipped"
    PARTIALLY_SHIPPED = "partially_shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


ACTIVE_ORDER_STATUSES = frozenset(
    {OrderStatus.PLACED, OrderStatus.SHIPPED, OrderStatus.PARTIALLY_SHIPPED}
)


class ShipmentStatus(StrEnum):
    LABEL_CREATED = "label_created"
    IN_TRANSIT = "in_transit"
    OUT_FOR_DELIVERY = "out_for_delivery"
    DELIVERED = "delivered"
    LOST = "lost"
    RETURNED_TO_SENDER = "returned_to_sender"


class ShipmentEventCode(StrEnum):
    LABEL_CREATED = "label_created"
    PICKED_UP = "picked_up"
    IN_TRANSIT = "in_transit"
    OUT_FOR_DELIVERY = "out_for_delivery"
    DELIVERED = "delivered"
    DELIVERY_ATTEMPTED = "delivery_attempted"
    EXCEPTION = "exception"


class ReturnReason(StrEnum):
    DAMAGED = "damaged"
    WRONG_ITEM = "wrong_item"
    NOT_AS_DESCRIBED = "not_as_described"
    NO_LONGER_NEEDED = "no_longer_needed"


class ReturnStatus(StrEnum):
    REQUESTED = "requested"
    LABEL_ISSUED = "label_issued"
    RECEIVED = "received"
    REJECTED = "rejected"
    COMPLETED = "completed"


class RefundStatus(StrEnum):
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class RefundReason(StrEnum):
    RETURN = "return"
    CANCELLATION = "cancellation"
    MISSING_DELIVERY = "missing_delivery"
    GOODWILL = "goodwill"


class TicketCategory(StrEnum):
    MISSING_DELIVERY = "missing_delivery"
    DAMAGED_PRODUCT = "damaged_product"
    RETURN_REFUND = "return_refund"
    ORDER_STATUS = "order_status"
    OTHER = "other"


class TicketStatus(StrEnum):
    OPEN = "open"
    PENDING_CUSTOMER = "pending_customer"
    ESCALATED = "escalated"
    RESOLVED = "resolved"


class TicketPriority(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class ConversationChannel(StrEnum):
    VOICE = "voice"
    TEXT = "text"


class ConversationOutcome(StrEnum):
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    ABANDONED = "abandoned"


class TurnRole(StrEnum):
    CUSTOMER = "customer"
    ASSISTANT = "assistant"
    TOOL = "tool"
    SYSTEM = "system"
