(function () {
    "use strict";

    const FIELD_META = {
        soil_moisture: { label: "Soil moisture", unit: "%", digits: 1 },
        temperature: { label: "Air temperature", unit: "°C", digits: 1 },
        humidity: { label: "Air humidity", unit: "%", digits: 1 },
        ph: { label: "Soil pH", unit: "", digits: 2 },
        nitrogen: { label: "Nitrogen", unit: "sensor units", digits: 1 },
        phosphorus: { label: "Phosphorus", unit: "sensor units", digits: 1 },
        potassium: { label: "Potassium", unit: "sensor units", digits: 1 },
        rainfall: { label: "Rainfall", unit: "mm", digits: 1 }
    };

    let weeklyData = null;
    let latestReading = null;
    let refreshInProgress = false;

    function numberOrNull(value) {
        if (value === null || value === undefined || value === "") return null;
        const number = Number(value);
        return Number.isFinite(number) ? number : null;
    }

    function formatValue(value, field) {
        const number = numberOrNull(value);
        if (number === null) return "No reading";
        const meta = FIELD_META[field];
        return number.toFixed(meta.digits) + (meta.unit ? " " + meta.unit : "");
    }

    function parseTimestamp(value) {
        if (!value) return null;
        const normalized = String(value).replace(" ", "T");
        const hasZone = /(?:Z|[+-]\d\d:\d\d)$/.test(normalized);
        const parsed = new Date(hasZone ? normalized : normalized + "Z");
        return Number.isNaN(parsed.getTime()) ? null : parsed;
    }

    function formatTime(value, includeDate) {
        const date = parseTimestamp(value);
        if (!date) return "Time unavailable";
        const options = includeDate
            ? { day: "numeric", month: "short", hour: "numeric", minute: "2-digit" }
            : { hour: "numeric", minute: "2-digit" };
        return new Intl.DateTimeFormat(undefined, options).format(date);
    }

    function setConnectionState(kind, label) {
        const pill = document.getElementById("monitor-live");
        pill.className = "monitor-live " + kind;
        pill.lastChild.textContent = " " + label;
    }

    function renderSummary(data, latest) {
        document.getElementById("weekly-condition").textContent = data.condition || "No recent data";
        document.getElementById("weekly-coverage").textContent =
            (data.days_with_readings || 0) + " of the last 7 days with a sensor packet";
        document.getElementById("weekly-reading-count").textContent =
            Number(data.reading_count || 0).toLocaleString();
        document.getElementById("weekly-days-count").textContent =
            Number(data.days_with_readings || 0) + " / 7";

        const stamp = latest && latest.timestamp;
        document.getElementById("last-reading-time").textContent =
            stamp ? formatTime(stamp, true) : "Waiting";
        const ageLabel = document.getElementById("last-reading-age");
        const date = parseTimestamp(stamp);
        if (!date) {
            ageLabel.textContent = "No sensor packet received yet";
            setConnectionState("is-waiting", "Waiting for sensor data");
        } else {
            const minutes = Math.max(0, Math.floor((Date.now() - date.getTime()) / 60000));
            if (minutes < 1) ageLabel.textContent = "Received just now";
            else if (minutes < 60) ageLabel.textContent = "Received " + minutes + " min ago";
            else if (minutes < 1440) ageLabel.textContent = "Received " + Math.floor(minutes / 60) + " hr ago";
            else ageLabel.textContent = "Received " + Math.floor(minutes / 1440) + " days ago";

            if (minutes > 10) {
                setConnectionState("is-stale", "No new packet in " + ageLabel.textContent.replace("Received ", ""));
            } else {
                setConnectionState("is-live", "Receiving sensor data");
            }
        }
    }

    function renderLatest(latest, fields) {
        const container = document.getElementById("latest-reading-grid");
        container.replaceChildren();
        Object.keys(FIELD_META).forEach(function (key) {
            const card = document.createElement("div");
            card.className = "monitor-reading-item";
            const label = document.createElement("span");
            label.textContent = FIELD_META[key].label;
            const value = document.createElement("strong");
            value.textContent = formatValue(latest && latest[key], key);
            if (numberOrNull(latest && latest[key]) === null) {
                value.classList.add("is-missing");
            }
            const summary = fields && fields[key];
            const weekly = document.createElement("small");
            weekly.textContent = summary && summary.samples
                ? summary.samples + " readings · weekly median " + formatValue(summary.weekly_median, key)
                : "No readings this week";
            card.append(label, value, weekly);
            container.appendChild(card);
        });
    }

    function renderActions(actions) {
        const container = document.getElementById("soil-actions");
        container.replaceChildren();
        (actions || []).forEach(function (action, index) {
            const card = document.createElement("article");
            card.className = "monitor-action-card priority-" + (action.priority || "steady");
            const marker = document.createElement("span");
            marker.className = "monitor-action-index";
            marker.textContent = String(index + 1).padStart(2, "0");
            const body = document.createElement("div");
            const title = document.createElement("strong");
            title.textContent = action.title || "Field check";
            const detail = document.createElement("p");
            detail.textContent = action.detail || "";
            body.append(title, detail);
            card.append(marker, body);
            container.appendChild(card);
        });
        if (!actions || actions.length === 0) {
            const empty = document.createElement("p");
            empty.className = "monitor-loading";
            empty.textContent = "No weekly recommendations are available yet.";
            container.appendChild(empty);
        }
    }

    function svgElement(name, attributes) {
        const element = document.createElementNS("http://www.w3.org/2000/svg", name);
        Object.keys(attributes || {}).forEach(function (key) {
            element.setAttribute(key, attributes[key]);
        });
        return element;
    }

    function renderChart() {
        if (!weeklyData) return;
        const metric = document.getElementById("metric-select").value;
        const meta = FIELD_META[metric];
        const svg = document.getElementById("sensor-chart");
        const empty = document.getElementById("chart-empty");
        const points = (weeklyData.readings || []).map(function (row) {
            return { timestamp: row.timestamp, value: numberOrNull(row[metric]) };
        });
        const usable = points.filter(function (point) { return point.value !== null; });

        document.getElementById("chart-metric-label").textContent = meta.label;
        document.getElementById("chart-sample-count").textContent =
            usable.length + (usable.length === 1 ? " reading" : " readings");
        document.getElementById("chart-latest-value").textContent =
            usable.length ? formatValue(usable[usable.length - 1].value, metric) : "—";

        svg.replaceChildren();
        svg.setAttribute("viewBox", "0 0 760 260");
        svg.setAttribute("preserveAspectRatio", "none");
        svg.setAttribute("aria-label", meta.label + " sensor readings over the last 7 days");
        empty.hidden = usable.length > 0;
        if (!usable.length) return;

        const width = 760;
        const height = 260;
        const left = 48;
        const right = 18;
        const top = 18;
        const bottom = 42;
        const plotWidth = width - left - right;
        const plotHeight = height - top - bottom;
        let min = Math.min.apply(null, usable.map(function (point) { return point.value; }));
        let max = Math.max.apply(null, usable.map(function (point) { return point.value; }));
        if (min === max) {
            const padding = Math.max(Math.abs(min) * 0.08, 1);
            min -= padding;
            max += padding;
        } else {
            const padding = (max - min) * 0.12;
            min -= padding;
            max += padding;
        }

        for (let grid = 0; grid <= 4; grid += 1) {
            const y = top + (plotHeight / 4) * grid;
            svg.appendChild(svgElement("line", {
                x1: left, y1: y, x2: width - right, y2: y, class: "chart-gridline"
            }));
            const label = svgElement("text", {
                x: left - 9, y: y + 4, class: "chart-axis-label", "text-anchor": "end"
            });
            label.textContent = (max - ((max - min) / 4) * grid).toFixed(1);
            svg.appendChild(label);
        }

        function xAt(index) {
            return left + (points.length < 2 ? plotWidth / 2 : (plotWidth * index) / (points.length - 1));
        }
        function yAt(value) {
            return top + plotHeight - ((value - min) / (max - min)) * plotHeight;
        }

        let pathSegment = "";
        const segments = [];
        points.forEach(function (point, index) {
            if (point.value === null) {
                if (pathSegment) segments.push(pathSegment);
                pathSegment = "";
                return;
            }
            pathSegment += (pathSegment ? " L " : "M ") + xAt(index) + " " + yAt(point.value);
        });
        if (pathSegment) segments.push(pathSegment);
        segments.forEach(function (path) {
            svg.appendChild(svgElement("path", { d: path, class: "chart-line" }));
        });
        points.forEach(function (point, index) {
            if (point.value === null) return;
            const circle = svgElement("circle", {
                cx: xAt(index), cy: yAt(point.value), r: 4, class: "chart-point"
            });
            const title = svgElement("title");
            title.textContent = formatTime(point.timestamp, true) + " · " + formatValue(point.value, metric);
            circle.appendChild(title);
            svg.appendChild(circle);
        });

        const firstTime = points.find(function (point) { return point.timestamp; });
        const lastTime = points.slice().reverse().find(function (point) { return point.timestamp; });
        [
            { x: left, anchor: "start", value: firstTime && firstTime.timestamp },
            { x: width - right, anchor: "end", value: lastTime && lastTime.timestamp }
        ].forEach(function (item) {
            if (!item.value) return;
            const label = svgElement("text", {
                x: item.x, y: height - 12, class: "chart-axis-label", "text-anchor": item.anchor
            });
            label.textContent = formatTime(item.value, true);
            svg.appendChild(label);
        });
    }

    async function refreshReadings() {
        if (refreshInProgress) return;
        refreshInProgress = true;
        const refreshButton = document.getElementById("refresh-monitor");
        refreshButton.disabled = true;
        try {
            const responses = await Promise.all([
                fetch("/api/sensors/week", { headers: { "Accept": "application/json" } }),
                fetch("/api/sensors/latest", { headers: { "Accept": "application/json" } })
            ]);
            if (responses.some(function (response) { return !response.ok; })) {
                throw new Error("Sensor data could not be loaded.");
            }
            const weekly = await responses[0].json();
            const latest = await responses[1].json();
            weeklyData = weekly.data || {};
            latestReading = latest.data || null;
            renderSummary(weeklyData, latestReading);
            renderLatest(latestReading, weeklyData.fields || {});
            renderActions(weeklyData.recommendations || []);
            document.getElementById("coverage-note").textContent =
                weeklyData.coverage_note ||
                "Soil pH and moisture checks are screening indicators. Crop stage, soil type and probe calibration affect interpretation; confirm nutrient and amendment decisions with a soil test and local agricultural advice.";
            renderChart();
        } catch (error) {
            setConnectionState("is-error", "Connection unavailable");
            document.getElementById("weekly-condition").textContent = "Could not refresh";
            document.getElementById("weekly-coverage").textContent = error.message;
        } finally {
            refreshInProgress = false;
            refreshButton.disabled = false;
        }
    }

    document.getElementById("refresh-monitor").addEventListener("click", refreshReadings);
    document.getElementById("metric-select").addEventListener("change", renderChart);
    refreshReadings();
    window.setInterval(function () {
        if (!document.hidden) refreshReadings();
    }, 60000);
    document.addEventListener("visibilitychange", function () {
        if (!document.hidden) refreshReadings();
    });
}());
