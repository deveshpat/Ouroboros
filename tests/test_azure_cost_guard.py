import pytest

from ouroboros.utils.azure_cost_guard import (
    AzureRetailPrice,
    build_retail_price_filter,
    estimate_cost,
    estimate_from_hourly_rate,
    select_linux_payg_vm_price,
)


def test_select_linux_payg_vm_price_ignores_windows_spot_and_low_priority():
    items = [
        {
            "armRegionName": "eastus",
            "armSkuName": "Standard_NC40ads_H100_v5",
            "productName": "Virtual Machines NCadsH100v5 Series Windows",
            "meterName": "NC40adsH100v5",
            "unitPrice": 8.82,
            "unitOfMeasure": "1 Hour",
        },
        {
            "armRegionName": "eastus",
            "armSkuName": "Standard_NC40ads_H100_v5",
            "productName": "Virtual Machines NCadsH100v5 Series",
            "meterName": "NC40adsH100v5 Spot",
            "unitPrice": 2.19,
            "unitOfMeasure": "1 Hour",
        },
        {
            "armRegionName": "eastus",
            "armSkuName": "Standard_NC40ads_H100_v5",
            "productName": "Virtual Machines NCadsH100v5 Series",
            "meterName": "NC40adsH100v5",
            "unitPrice": 6.98,
            "unitOfMeasure": "1 Hour",
        },
    ]

    price = select_linux_payg_vm_price(items, sku="Standard_NC40ads_H100_v5", region="eastus")

    assert price.unit_price_usd == 6.98
    assert price.product == "Virtual Machines NCadsH100v5 Series"
    assert price.meter == "NC40adsH100v5"


def test_estimate_cost_applies_safety_buffer_and_max_safe_hours():
    price = AzureRetailPrice(
        region="eastus",
        sku="Standard_NC40ads_H100_v5",
        meter="NC40adsH100v5",
        product="Virtual Machines NCadsH100v5 Series",
        unit_price_usd=6.98,
        unit="1 Hour",
    )

    estimate = estimate_cost(
        price=price,
        planned_hours=18.0,
        budget_usd=190.0,
        safety_buffer=0.25,
        overhead_usd=4.0,
    )

    assert estimate.base_compute_usd == pytest.approx(125.64)
    assert estimate.buffered_total_usd == pytest.approx(162.05)
    assert estimate.max_safe_hours == pytest.approx(21.2034, rel=1e-4)
    assert estimate.fits_budget is True


def test_estimate_cost_fails_when_buffered_total_exceeds_budget():
    price = AzureRetailPrice(
        region="eastus",
        sku="Standard_NC40ads_H100_v5",
        meter="NC40adsH100v5",
        product="Virtual Machines NCadsH100v5 Series",
        unit_price_usd=6.98,
        unit="1 Hour",
    )

    estimate = estimate_cost(price=price, planned_hours=24.0, budget_usd=190.0, safety_buffer=0.25)

    assert estimate.fits_budget is False


def test_estimate_from_hourly_rate_skips_retail_lookup():
    estimate = estimate_from_hourly_rate(
        region="eastus",
        sku="Standard_NC40ads_H100_v5",
        hourly_usd=6.98,
        planned_hours=18.75,
        budget_usd=190.0,
        safety_buffer=0.25,
        overhead_usd=4.0,
    )

    assert estimate.hourly_usd == pytest.approx(6.98)
    assert estimate.buffered_total_usd == pytest.approx(168.59375)
    assert estimate.fits_budget is True


def test_build_retail_price_filter_quotes_values():
    assert build_retail_price_filter(
        service_name="Virtual Machines",
        sku="Standard_NC40ads_H100_v5",
        region="eastus",
        price_type="Consumption",
    ) == (
        "serviceName eq 'Virtual Machines' and "
        "armSkuName eq 'Standard_NC40ads_H100_v5' and "
        "armRegionName eq 'eastus' and "
        "priceType eq 'Consumption'"
    )
