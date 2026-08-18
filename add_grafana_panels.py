import json
import shutil
from pathlib import Path

DASHBOARD = Path("grafana/dashboards/telco-churn-dashboard.json")
BACKUP = Path("grafana/dashboards/telco-churn-dashboard.backup.json")


def make_stat_panel(panel_id, title, expr, x, y, unit="short"):
    return {
        "id": panel_id,
        "type": "stat",
        "title": title,
        "gridPos": {
            "h": 6,
            "w": 6,
            "x": x,
            "y": y
        },
        "datasource": {
            "type": "prometheus",
            "uid": "prometheus"
        },
        "fieldConfig": {
            "defaults": {
                "unit": unit,
                "decimals": 2,
                "color": {
                    "mode": "thresholds"
                },
                "thresholds": {
                    "mode": "absolute",
                    "steps": [
                        {
                            "color": "green",
                            "value": None
                        },
                        {
                            "color": "red",
                            "value": 0
                        }
                    ]
                }
            },
            "overrides": []
        },
        "options": {
            "reduceOptions": {
                "values": False,
                "calcs": ["lastNotNull"],
                "fields": ""
            },
            "orientation": "auto",
            "textMode": "auto",
            "colorMode": "value",
            "graphMode": "area",
            "justifyMode": "auto"
        },
        "targets": [
            {
                "expr": expr,
                "refId": "A"
            }
        ]
    }


def make_time_series_panel(panel_id, title, expr, x, y, legend="{{prediction}}"):
    return {
        "id": panel_id,
        "type": "timeseries",
        "title": title,
        "gridPos": {
            "h": 8,
            "w": 12,
            "x": x,
            "y": y
        },
        "datasource": {
            "type": "prometheus",
            "uid": "prometheus"
        },
        "fieldConfig": {
            "defaults": {
                "color": {
                    "mode": "palette-classic"
                },
                "custom": {
                    "drawStyle": "line",
                    "lineInterpolation": "linear",
                    "lineWidth": 2,
                    "fillOpacity": 10,
                    "showPoints": "auto",
                    "spanNulls": False
                }
            },
            "overrides": []
        },
        "options": {
            "legend": {
                "displayMode": "list",
                "placement": "bottom"
            },
            "tooltip": {
                "mode": "multi",
                "sort": "desc"
            }
        },
        "targets": [
            {
                "expr": expr,
                "refId": "A",
                "legendFormat": legend
            }
        ]
    }


# ---------------------------------------------------------
# 1. Check dashboard exists
# ---------------------------------------------------------

if not DASHBOARD.exists():
    raise FileNotFoundError(
        f"Dashboard not found: {DASHBOARD}"
    )


# ---------------------------------------------------------
# 2. Create an additional safety backup
# ---------------------------------------------------------

shutil.copy2(DASHBOARD, BACKUP)

print("Backup created:")
print(BACKUP)


# ---------------------------------------------------------
# 3. Load dashboard
# ---------------------------------------------------------

with open(DASHBOARD, "r", encoding="utf-8") as f:
    dashboard = json.load(f)


# ---------------------------------------------------------
# 4. Find panels
# ---------------------------------------------------------

panels = dashboard.get("panels", [])

existing_titles = {
    panel.get("title", "").strip()
    for panel in panels
}


# ---------------------------------------------------------
# 5. Find next available panel ID
# ---------------------------------------------------------

existing_ids = []

for panel in panels:
    if isinstance(panel.get("id"), int):
        existing_ids.append(panel["id"])

next_id = max(existing_ids, default=0) + 1


# ---------------------------------------------------------
# 6. Find bottom of existing dashboard
# ---------------------------------------------------------

max_y = 0

for panel in panels:
    grid = panel.get("gridPos", {})
    y = grid.get("y", 0)
    h = grid.get("h", 0)

    max_y = max(max_y, y + h)


# Start new panels below existing panels
start_y = max_y + 1


# ---------------------------------------------------------
# 7. Panels to automatically add
# ---------------------------------------------------------

new_panels = []


# Row 1
definitions = [
    (
        "Successful Predictions",
        "prediction_success_total",
        "short"
    ),
    (
        "Failed Predictions",
        "prediction_failure_total",
        "short"
    ),
    (
        "API Availability",
        "api_up",
        "short"
    ),
    (
        "Model Loaded",
        "model_loaded",
        "short"
    ),
]

x_positions = [0, 6, 12, 18]

for i, (title, expr, unit) in enumerate(definitions):

    if title not in existing_titles:

        new_panels.append(
            make_stat_panel(
                next_id,
                title,
                expr,
                x_positions[i],
                start_y,
                unit
            )
        )

        next_id += 1


# Row 2
row2_y = start_y + 7

definitions = [
    (
        "Preprocessor Loaded",
        "preprocessor_loaded",
        "short"
    ),
    (
        "Model Confidence",
        "model_confidence",
        "percentunit"
    ),
    (
        "Input Feature Count",
        "input_feature_count",
        "short"
    ),
]

x_positions = [0, 6, 12]

for i, (title, expr, unit) in enumerate(definitions):

    if title not in existing_titles:

        new_panels.append(
            make_stat_panel(
                next_id,
                title,
                expr,
                x_positions[i],
                row2_y,
                unit
            )
        )

        next_id += 1


# Row 3
row3_y = row2_y + 7

if "Prediction Distribution" not in existing_titles:

    new_panels.append(
        make_time_series_panel(
            next_id,
            "Prediction Distribution",
            "sum by (prediction) (model_predictions_total)",
            0,
            row3_y,
            'prediction="{{prediction}}"'
        )
    )

    next_id += 1


# ---------------------------------------------------------
# 8. Add panels
# ---------------------------------------------------------

panels.extend(new_panels)

dashboard["panels"] = panels


# ---------------------------------------------------------
# 9. Save dashboard
# ---------------------------------------------------------

with open(DASHBOARD, "w", encoding="utf-8") as f:
    json.dump(
        dashboard,
        f,
        indent=2
    )


# ---------------------------------------------------------
# 10. Report
# ---------------------------------------------------------

print()
print("=" * 60)
print("GRAFANA DASHBOARD UPDATED")
print("=" * 60)
print(f"Existing panels: {len(panels) - len(new_panels)}")
print(f"New panels added: {len(new_panels)}")
print(f"Total panels: {len(panels)}")
print()

for panel in new_panels:
    print(f"  + {panel['title']}")

print()
print(f"Dashboard: {DASHBOARD}")
print(f"Backup:    {BACKUP}")
print("=" * 60)
