Office.initialize = () => {
  const item = Office.context.mailbox.item;

  // Only run in read mode for email
  if (item && item.itemType === Office.MailboxEnums.ItemType.Message) {
    // Get the email body
    item.body.getAsync("text", (result) => {
      if (result.status === Office.AsyncResultStatus.Succeeded) {
        const emailText = result.value;

        // Simulate model prediction (replace with your actual model call)
        const isPhishing = emailText.toLowerCase().includes("password") ||
                           emailText.toLowerCase().includes("urgent") ||
                           Math.random() > 0.5; // fallback randomness

        // Create a floating circle indicator
        const indicator = document.createElement("div");
        indicator.style.position = "fixed";
        indicator.style.top = "15px";
        indicator.style.right = "20px";
        indicator.style.width = "15px";
        indicator.style.height = "15px";
        indicator.style.borderRadius = "50%";
        indicator.style.zIndex = "9999";
        indicator.style.boxShadow = "0 0 5px rgba(0,0,0,0.3)";
        indicator.style.backgroundColor = isPhishing ? "red" : "green";
        indicator.title = isPhishing
          ? "⚠️ Phishing suspected!"
          : "✅ Safe email";

        document.body.appendChild(indicator);

        console.log("Phishing detector: injected indicator", isPhishing);
      } else {
        console.error("Could not read email body:", result.error);
      }
    });
  }
};
