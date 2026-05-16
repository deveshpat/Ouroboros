"""Azure cost preflight helpers for budget-bounded Ouroboros training."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence


AZURE_RETAIL_PRICES_URL = "https://prices.azure.com/api/retail/prices"
DEFAULT_H100_SKU = "Standard_NC40ads_H100_v5"
DEFAULT_BUDGET_USD = 190.0
DEFAULT_SAFETY_BUFFER = 0.25


@dataclass(frozen=True)
class AzureRetailPrice:
    region: str
    sku: str
    meter: str
    product: str
    unit_price_usd: float
    unit: str


@dataclass(frozen=True)
class AzureCostEstimate:
    region: str
    sku: str
    planned_hours: float
    instances: int
    hourly_usd: float
    base_compute_usd: float
    overhead_usd: float
    safety_buffer: float
    buffered_total_usd: float
    budget_usd: float
    max_safe_hours: float
    fits_budget: bool


def _odata_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def build_retail_price_filter(*, service_name: str, sku: str, region: str, price_type: str) -> str:
    return " and ".join(
        [
            f"serviceName eq {_odata_quote(service_name)}",
            f"armSkuName eq {_odata_quote(sku)}",
            f"armRegionName eq {_odata_quote(region)}",
            f"priceType eq {_odata_quote(price_type)}",
        ]
    )


def fetch_retail_price_items(
    *,
    sku: str,
    region: str,
    service_name: str = "Virtual Machines",
    price_type: str = "Consumption",
    timeout_seconds: float = 30.0,
) -> list[Mapping[str, Any]]:
    """Fetch Azure Retail Prices API rows for a VM SKU and region."""
    params = {
        "$filter": build_retail_price_filter(
            service_name=service_name,
            sku=sku,
            region=region,
            price_type=price_type,
        )
    }
    url = f"{AZURE_RETAIL_PRICES_URL}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return list(payload.get("Items") or [])


def _is_linux_payg_vm_meter(item: Mapping[str, Any]) -> bool:
    product = str(item.get("productName") or "")
    meter = str(item.get("meterName") or "")
    if "Windows" in product:
        return False
    lowered_meter = meter.lower()
    if "spot" in lowered_meter or "low priority" in lowered_meter:
        return False
    return True


def select_linux_payg_vm_price(items: Iterable[Mapping[str, Any]], *, sku: str, region: str) -> AzureRetailPrice:
    """Select the normal Linux pay-as-you-go VM price from Retail API rows."""
    candidates = [item for item in items if _is_linux_payg_vm_meter(item)]
    if not candidates:
        raise ValueError(f"No Linux pay-as-you-go price found for {sku} in {region}")
    if len(candidates) > 1:
        candidates.sort(key=lambda item: str(item.get("meterName") or ""))
    item = candidates[0]
    return AzureRetailPrice(
        region=str(item.get("armRegionName") or region),
        sku=str(item.get("armSkuName") or sku),
        meter=str(item.get("meterName") or ""),
        product=str(item.get("productName") or ""),
        unit_price_usd=float(item["unitPrice"]),
        unit=str(item.get("unitOfMeasure") or "1 Hour"),
    )


def estimate_cost(
    *,
    price: AzureRetailPrice,
    planned_hours: float,
    budget_usd: float = DEFAULT_BUDGET_USD,
    safety_buffer: float = DEFAULT_SAFETY_BUFFER,
    instances: int = 1,
    overhead_usd: float = 0.0,
) -> AzureCostEstimate:
    if planned_hours <= 0:
        raise ValueError("planned_hours must be positive")
    if budget_usd <= 0:
        raise ValueError("budget_usd must be positive")
    if safety_buffer < 0:
        raise ValueError("safety_buffer must be non-negative")
    if instances <= 0:
        raise ValueError("instances must be positive")
    if overhead_usd < 0:
        raise ValueError("overhead_usd must be non-negative")

    hourly = price.unit_price_usd * instances
    base = hourly * planned_hours
    buffered = (base + overhead_usd) * (1.0 + safety_buffer)
    max_safe_hours = max((budget_usd / (1.0 + safety_buffer) - overhead_usd) / hourly, 0.0)
    return AzureCostEstimate(
        region=price.region,
        sku=price.sku,
        planned_hours=float(planned_hours),
        instances=int(instances),
        hourly_usd=float(hourly),
        base_compute_usd=float(base),
        overhead_usd=float(overhead_usd),
        safety_buffer=float(safety_buffer),
        buffered_total_usd=float(buffered),
        budget_usd=float(budget_usd),
        max_safe_hours=float(max_safe_hours),
        fits_budget=buffered <= budget_usd,
    )


def estimate_from_retail_api(
    *,
    region: str,
    sku: str = DEFAULT_H100_SKU,
    planned_hours: float,
    budget_usd: float = DEFAULT_BUDGET_USD,
    safety_buffer: float = DEFAULT_SAFETY_BUFFER,
    instances: int = 1,
    overhead_usd: float = 0.0,
) -> AzureCostEstimate:
    items = fetch_retail_price_items(sku=sku, region=region)
    price = select_linux_payg_vm_price(items, sku=sku, region=region)
    return estimate_cost(
        price=price,
        planned_hours=planned_hours,
        budget_usd=budget_usd,
        safety_buffer=safety_buffer,
        instances=instances,
        overhead_usd=overhead_usd,
    )


def estimate_from_hourly_rate(
    *,
    region: str,
    hourly_usd: float,
    sku: str = DEFAULT_H100_SKU,
    planned_hours: float,
    budget_usd: float = DEFAULT_BUDGET_USD,
    safety_buffer: float = DEFAULT_SAFETY_BUFFER,
    instances: int = 1,
    overhead_usd: float = 0.0,
) -> AzureCostEstimate:
    if hourly_usd <= 0:
        raise ValueError("hourly_usd must be positive")
    price = AzureRetailPrice(
        region=region,
        sku=sku,
        meter=f"{sku} override",
        product="Manual hourly price override",
        unit_price_usd=float(hourly_usd),
        unit="1 Hour",
    )
    return estimate_cost(
        price=price,
        planned_hours=planned_hours,
        budget_usd=budget_usd,
        safety_buffer=safety_buffer,
        instances=instances,
        overhead_usd=overhead_usd,
    )


def _format_human(estimate: AzureCostEstimate) -> str:
    status = "PASS" if estimate.fits_budget else "FAIL"
    return "\n".join(
        [
            f"Azure cost preflight: {status}",
            f"  region: {estimate.region}",
            f"  sku: {estimate.sku}",
            f"  hourly: ${estimate.hourly_usd:.4f}/hr",
            f"  planned_hours: {estimate.planned_hours:.2f}",
            f"  base_compute: ${estimate.base_compute_usd:.2f}",
            f"  overhead: ${estimate.overhead_usd:.2f}",
            f"  safety_buffer: {estimate.safety_buffer:.0%}",
            f"  buffered_total: ${estimate.buffered_total_usd:.2f}",
            f"  budget: ${estimate.budget_usd:.2f}",
            f"  max_safe_hours: {estimate.max_safe_hours:.2f}",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Estimate Azure VM training cost before launch.")
    parser.add_argument("--region", required=True)
    parser.add_argument("--sku", default=DEFAULT_H100_SKU)
    parser.add_argument("--planned_hours", type=float, required=True)
    parser.add_argument(
        "--hourly_usd",
        type=float,
        default=None,
        help="Skip Retail Prices API and use this explicit hourly USD rate.",
    )
    parser.add_argument("--budget_usd", type=float, default=DEFAULT_BUDGET_USD)
    parser.add_argument("--safety_buffer", type=float, default=DEFAULT_SAFETY_BUFFER)
    parser.add_argument("--instances", type=int, default=1)
    parser.add_argument("--overhead_usd", type=float, default=0.0)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--require_budget_fit", action="store_true", help="Exit 2 if estimate exceeds budget.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.hourly_usd is not None:
        estimate = estimate_from_hourly_rate(
            region=args.region,
            sku=args.sku,
            hourly_usd=args.hourly_usd,
            planned_hours=args.planned_hours,
            budget_usd=args.budget_usd,
            safety_buffer=args.safety_buffer,
            instances=args.instances,
            overhead_usd=args.overhead_usd,
        )
    else:
        estimate = estimate_from_retail_api(
            region=args.region,
            sku=args.sku,
            planned_hours=args.planned_hours,
            budget_usd=args.budget_usd,
            safety_buffer=args.safety_buffer,
            instances=args.instances,
            overhead_usd=args.overhead_usd,
        )
    if args.json:
        print(json.dumps(asdict(estimate), indent=2, sort_keys=True))
    else:
        print(_format_human(estimate))
    if args.require_budget_fit and not estimate.fits_budget:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
