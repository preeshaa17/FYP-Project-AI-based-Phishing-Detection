async function onMessageReceived(event) {
  try {
    const item = Office.context.mailbox.item;

    // Get email body (plain text)
    item.body.getAsync("text", async (result) => {
      if (result.status === Office.AsyncResultStatus.Succeeded) {
        const emailBody = result.value;

        // Send to FastAPI backend for prediction
        const response = await fetch("https://your-api.onrender.com/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: emailBody })
        });

        const data = await response.json();

        // Pass result to UI display (taskpane.js)
        Office.context.ui.messageParent(JSON.stringify(data));
      }
    });
  } catch (error) {
    console.error("Error scanning email:", error);
  } finally {
    event.completed(); // important!
  }
}
