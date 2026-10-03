(function () {
    "use strict";

    const UNITS = {
        soil_moisture: "%",
        ph: "",
        temperature: "°C",
        humidity: "%",
        nitrogen: "",
        phosphorus: "",
        potassium: ""
    };

    function displayValue(value, field) {
        if (value === null || value === undefined || value === "" ||
            String(value).toLowerCase() === "not available") {
            return "—";
        }
        const numeric = Number(value);
        if (!Number.isFinite(numeric)) return "—";
        const precision = field === "ph" ? 2 : 1;
        return numeric.toFixed(precision) + (UNITS[field] ? " " + UNITS[field] : "");
    }

    function humanTime(value) {
        if (!value) return "No sensor packet received yet";
        const normalized = String(value).replace(" ", "T");
        const date = new Date(/(?:Z|[+-]\d\d:\d\d)$/.test(normalized) ? normalized : normalized + "Z");
        if (Number.isNaN(date.getTime())) return "Sensor time unavailable";
        return "Last packet: " + new Intl.DateTimeFormat(undefined, {
            day: "numeric", month: "short", hour: "numeric", minute: "2-digit"
        }).format(date);
    }

    async function refreshDashboard() {
        const status = document.getElementById("dashboard-sync-status");
        try {
            const response = await fetch("/api/sensors/latest", {
                headers: { "Accept": "application/json" }
            });
            if (!response.ok) throw new Error("Sensor status unavailable");
            const result = await response.json();
            const reading = result.data;
            Object.keys(UNITS).forEach(function (field) {
                const element = document.querySelector('[data-sensor-value="' + field + '"]');
                if (element) element.textContent = displayValue(reading && reading[field], field);
            });
            document.getElementById("dashboard-reading-time").textContent =
                humanTime(reading && reading.timestamp);

            if (!reading || !reading.timestamp) {
                status.textContent = "Waiting for a sensor packet";
                status.className = "dashboard-sync-status is-waiting";
                return;
            }
            const stamp = new Date(String(reading.timestamp).replace(" ", "T") +
                (/(?:Z|[+-]\d\d:\d\d)$/.test(reading.timestamp) ? "" : "Z"));
            const minutes = Math.max(0, Math.floor((Date.now() - stamp.getTime()) / 60000));
            if (minutes > 10) {
                status.textContent = "No new packet in " + Math.floor(minutes / 60) + " hr";
                status.className = "dashboard-sync-status is-stale";
            } else {
                status.textContent = "Receiving sensor data";
                status.className = "dashboard-sync-status is-live";
            }
        } catch (error) {
            status.textContent = "Sensor connection unavailable";
            status.className = "dashboard-sync-status is-stale";
        }
    }

    refreshDashboard();
    window.setInterval(function () {
        if (!document.hidden) refreshDashboard();
    }, 60000);
}());
