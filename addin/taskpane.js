Office.onReady(() => {
  Office.context.ui.addHandlerAsync(
    Office.EventType.DialogParentMessageReceived,
    (arg) => {
      const data = JSON.parse(arg.message);
      const resultBox = document.getElementById("result");

      if (data.prediction === "phishing") {
        resultBox.innerText = "⚠️ Phishing Detected!";
        resultBox.className = "red";
      } else if (data.prediction === "legitimate") {
        resultBox.innerText = "✅ Safe Email";
        resultBox.className = "green";
      } else {
        resultBox.innerText = "No prediction available";
        resultBox.className = "";
      }
    }
  );
});
