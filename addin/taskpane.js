Office.onReady(() => {
  document.getElementById("scanButton").onclick = async function () {
    Office.context.mailbox.item.body.getAsync("text", async (result) => {
      if (result.status === Office.AsyncResultStatus.Succeeded) {
        const emailBody = result.value;

        // Send to FastAPI backend for prediction
        const response = await fetch("https://your-api-url.onrender.com/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: emailBody }),
        });

        const data = await response.json();
        document.getElementById("result").innerText =
          `Prediction: ${data.prediction}`;
      }
    });
  };
});
