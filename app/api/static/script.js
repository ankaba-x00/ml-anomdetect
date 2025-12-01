// -------- Load country list dynamically --------
async function loadCountries() {
    try {
        const res = await fetch("/api/countries");
        const data = await res.json();

        const sel = document.getElementById("country");
        sel.innerHTML = "";
        data.countries.forEach(c => {
            const opt = document.createElement("option");
            opt.value = c;
            opt.textContent = c;
            sel.appendChild(opt);
        });
    } catch (e) {
        console.error("[ERROR] Could not load countries", e);
    }
}

loadCountries();

// -------- Check date --------
function validateDate(dateStr) {
    if (!dateStr) return { ok: false, msg: "Please select a date." };

    const selected = new Date(dateStr);
    if (isNaN(selected.getTime())) {
        return { ok: false, msg: "Invalid date format." };
    }

    const MIN = new Date("2025-11-14T00:00:00Z");

    const today = new Date();
    const MAX = new Date(today.getFullYear(), today.getMonth(), today.getDate() - 1);

    if (selected < MIN) {
        return { ok: false, msg: "Date must be on or after 11/14/2025." };
    }
    if (selected > MAX) {
        return { ok: false, msg: "Date cannot be today or in the future." };
    }
    return { ok: true };
}


// -------- Main inference function --------
async function runInference() {
    const country = document.getElementById("country").value;
    const date    = document.getElementById("date").value;
    const out     = document.getElementById("result");
    const loading = document.getElementById("loading");

    const check = validateDate(date);
    if (!check.ok) {
        out.style.display = "block";
        out.className = "result-box bad";
        out.innerHTML = check.msg;
        loading.innerHTML = "";
        return;
    }

    loading.innerHTML = "Running detection...";
    out.style.display = "none";

    try {
        const res = await fetch("/api/infer", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                country: country,
                date_from: date,
                date_to: date
            })
        });

        const data = await res.json();
        loading.innerHTML = "";

        if (!res.ok) {
            out.style.display = "block";
            out.className = "result-box bad";
            out.innerHTML = `Error: ${data.detail}`;
            return;
        }

        // Display detection result
        out.style.display = "block";

        if (data.detected > 0) {
            out.className = "result-box bad";
            out.innerHTML = `<b>${data.detected}</b> anomalies detected<br><small>${data.anomalies.join(", ")}</small>`;
        } else {
            out.className = "result-box good";
            out.innerHTML = `No anomalies detected`;
        }

    } catch (err) {
        loading.innerHTML = "";
        out.style.display = "block";
        out.className = "result-box bad";
        out.innerHTML = "Inference failed.";
        console.error(err);
    }
}