function onMessageReceived(event) {
  Office.context.mailbox.item.body.getAsync("text", async (result) => {
    if (result.status === Office.AsyncResultStatus.Succeeded) {
      const emailBody = result.value;

      // Send email content to backend
      const response = await fetch("https://your-api.onrender.com/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: emailBody })
      });

      const data = await response.json();

      // Update UI
      const resultElement = document.getElementById("result");
      if (data.prediction === "phishing") {
        resultElement.innerHTML = "⚠️ Phishing Detected!";
        resultElement.style.backgroundColor = "red";
      } else {
        resultElement.innerHTML = "✅ Safe Email";
        resultElement.style.backgroundColor = "green";
      }
    }
  });

  event.completed(); // Important to signal event completion
}
