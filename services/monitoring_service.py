"""Conservative, explainable checks for weekly farm sensor readings."""

from datetime import datetime, timedelta, timezone


FIELD_LABELS = {
    "soil_moisture": ("Soil moisture", "%"),
    "temperature": ("Air temperature", "°C"),
    "humidity": ("Air humidity", "%"),
    "ph": ("Soil pH", ""),
    "nitrogen": ("Nitrogen", "sensor units"),
    "phosphorus": ("Phosphorus", "sensor units"),
    "potassium": ("Potassium", "sensor units"),
    "rainfall": ("Rainfall", "mm"),
}


def _number(value):
    if value is None or str(value).strip().lower() in {
        "", "none", "null", "not available", "n/a"
    }:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _timestamp(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if not ordered:
        return None
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _sort_timestamp(value):
    stamp = _timestamp(value)
    if stamp is None:
        return float("-inf")
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.timestamp()


def analyze_soil_week(readings):
    """Summarize the last week without inventing missing sensor readings."""
    rows = sorted(
        (dict(row) for row in (readings or [])),
        key=lambda row: (_sort_timestamp(row.get("timestamp")), row.get("id", 0)),
    )
    now = datetime.now().astimezone()
    cutoff = now - timedelta(days=7)
    recent = []
    for row in rows:
        stamp = _timestamp(row.get("timestamp"))
        if stamp is None:
            continue
        # SQLite CURRENT_TIMESTAMP values are UTC but do not carry a zone.
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        stamp = stamp.astimezone(now.tzinfo)
        if stamp >= cutoff:
            row["timestamp"] = stamp.isoformat()
            recent.append(row)

    fields = {}
    for key, (label, unit) in FIELD_LABELS.items():
        points = [
            {"timestamp": row["timestamp"], "value": value}
            for row in recent
            if (value := _number(row.get(key))) is not None
        ]
        values = [point["value"] for point in points]
        latest = points[-1]["value"] if points else None
        fields[key] = {
            "label": label,
            "unit": unit,
            "latest": latest,
            "samples": len(points),
            "weekly_median": _median(values),
            "weekly_minimum": min(values) if values else None,
            "weekly_maximum": max(values) if values else None,
        }

    recommendations = []
    ph_values = [
        value for row in recent
        if (value := _number(row.get("ph"))) is not None
    ]
    moisture_values = [
        value for row in recent
        if (value := _number(row.get("soil_moisture"))) is not None
    ]
    nutrient_fields = ("nitrogen", "phosphorus", "potassium")
    missing_fields = [
        FIELD_LABELS[key][0]
        for key in ("soil_moisture", "ph", *nutrient_fields)
        if fields[key]["samples"] == 0
    ]

    if not recent:
        recommendations.append({
            "priority": "setup",
            "title": "Connect your field sensors",
            "detail": "No readings were received in the last 7 days. Connect a sensor or send a reading to begin the weekly soil review.",
        })
    else:
        if ph_values and _median(ph_values) < 5.5:
            recommendations.append({
                "priority": "attention",
                "title": "Confirm the acidic pH readings",
                "detail": "The weekly median is below the broad screening band of pH 5.5–7.5. Confirm with a composite soil test and get crop-specific advice from your local KVK before applying amendments.",
            })
        elif ph_values and _median(ph_values) > 7.5:
            recommendations.append({
                "priority": "attention",
                "title": "Confirm the alkaline pH readings",
                "detail": "The weekly median is above the broad screening band of pH 5.5–7.5. Confirm with a composite soil test; the right amendment depends on the crop and whether the soil is sodic.",
            })

        if moisture_values:
            recent_moisture = _median(moisture_values[-min(10, len(moisture_values)):])
            if recent_moisture < 25:
                recommendations.append({
                    "priority": "attention",
                    "title": "Check whether the root zone is drying",
                    "detail": "Recent soil-moisture readings are low. Verify the probe against the soil at root depth, then inspect irrigation and rainfall before deciding whether to water.",
                })
            elif recent_moisture > 85:
                recommendations.append({
                    "priority": "attention",
                    "title": "Check for persistent saturation",
                    "detail": "Recent soil-moisture readings are high. Check drainage and confirm the probe is not sitting in pooled water to reduce waterlogging risk.",
                })

        if missing_fields:
            recommendations.append({
                "priority": "data",
                "title": "Fill the soil-data gaps",
                "detail": "No weekly readings were received for " + ", ".join(missing_fields) + ". Air humidity is not a substitute for soil moisture, and nutrient levels need a soil test or a calibrated sensor.",
            })

        if not any(item["priority"] == "attention" for item in recommendations):
            recommendations.append({
                "priority": "steady",
                "title": "Keep the weekly checks going",
                "detail": "No broad pH or moisture flag was found in the available readings. This is a screening check, not a crop-specific soil diagnosis; use a soil test for nutrient and amendment decisions.",
            })

    dates = []
    for row in recent:
        date = row["timestamp"][:10]
        if date not in dates:
            dates.append(date)

    if not recent:
        condition = "No recent data"
    elif len(dates) < 3:
        condition = "Limited weekly coverage"
    elif any(item["priority"] == "attention" for item in recommendations):
        condition = "Needs a field check"
    else:
        condition = "No broad flags found"

    serializable_rows = []
    for row in recent[-500:]:
        serializable_rows.append({
            "id": row.get("id"),
            "timestamp": row["timestamp"],
            **{key: _number(row.get(key)) for key in FIELD_LABELS},
        })

    return {
        "period_days": 7,
        "reading_count": len(recent),
        "days_with_readings": len(dates),
        "condition": condition,
        "fields": fields,
        "recommendations": recommendations,
        "readings": serializable_rows,
        "coverage_note": "Moisture bands depend on soil texture, crop stage, probe placement and calibration. NPK values are shown as trends only; use a soil test for fertilizer decisions.",
    }
