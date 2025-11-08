Office.onReady(info => {
  if (info.host === Office.HostType.Outlook) {
    console.log("AI Phishing Detector Add-in loaded");

    const scanBtn = document.getElementById("scanBtn");
    const resultBox = document.getElementById("result");

    scanBtn.addEventListener("click", async () => {
      try {
        const item = Office.context.mailbox.item;
        const subject = item.subject || "No subject";
        const sender = item.from ? item.from.emailAddress : "Unknown sender";

        resultBox.innerText = "Scanning email...";
        resultBox.className = "status-box";

        // Get body text (async)
        item.body.getAsync("text", async (res) => {
          if (res.status === Office.AsyncResultStatus.Succeeded) {
            const emailBody = res.value;

            // ✅ Replace this with your actual backend API later
            // Example: const response = await fetch("https://localhost:5000/predict", { method: "POST", body: JSON.stringify({ text: emailBody }) });
            // const data = await response.json();

            // Simulated prediction for now:
            const prediction = emailBody.includes("password") ? "phishing" : "legitimate";

            if (prediction === "phishing") {
              resultBox.innerText = `⚠️ Phishing Detected!\nFrom: ${sender}\nSubject: ${subject}`;
              resultBox.className = "status-box red";
            } else {
              resultBox.innerText = `✅ Safe Email\nFrom: ${sender}\nSubject: ${subject}`;
              resultBox.className = "status-box green";
            }
          } else {
            resultBox.innerText = "❌ Could not read email content.";
            resultBox.className = "status-box red";
          }
        });
      } catch (err) {
        console.error(err);
        resultBox.innerText = "❌ Error running detection.";
        resultBox.className = "status-box red";
      }
    });
  }
});
