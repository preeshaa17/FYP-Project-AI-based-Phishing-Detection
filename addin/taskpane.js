// --- taskpane.js (Final version for API integration) ---

// 📢 CONFIGURATION: MUST match the settings in your main.py
const API_URL = "https://localhost:5000/predict"; 
const API_KEY = "change-me"; // ⚠️ CRITICAL: MUST MATCH THE KEY SET IN main.py

Office.onReady(info => {
  if (info.host === Office.HostType.Outlook) {
    console.log("AI Phishing Detector Task Pane loaded and ready to connect to API.");

    const scanBtn = document.getElementById("scanBtn");
    const resultBox = document.getElementById("result");
    
    // Helper function to wrap Office.js getAsync calls in a Promise
    const getBodyContent = (format) => new Promise((resolve) => {
        const item = Office.context.mailbox.item;
        item.body.getAsync(format, (res) => {
            // Resolve with the value or an empty string on failure
            resolve(res.status === Office.AsyncResultStatus.Succeeded ? res.value : "");
        });
    });

    // Simple placeholder to extract URLs from text
    const extractUrls = (text) => {
        const urlRegex = /(https?:\/\/[^\s"]+)/g;
        // Basic extraction, limits to first 10 URLs
        return (text.match(urlRegex) || []).slice(0, 10);
    };


    scanBtn.addEventListener("click", async () => {
      try {
        const item = Office.context.mailbox.item;
        const subject = item.subject || "";
        const sender = item.from ? item.from.emailAddress : "";
        
        resultBox.innerText = "Scanning email and connecting to local API...";
        resultBox.className = "status-box";

        // 1. Fetch Body Content (runs both async and waits for both)
        const [bodyText, bodyHtml] = await Promise.all([
            getBodyContent("text"),
            getBodyContent("html")
        ]);

        if (!bodyText && !bodyHtml) {
             resultBox.innerText = "❌ Could not read email content. Ensure the item is a message.";
             resultBox.className = "status-box red";
             return;
        }
        
        // 2. Extract Features
        const urls = extractUrls(bodyText);

        // 3. Prepare the Payload (Must match Pydantic model 'EmailIn' in main.py)
        const payload = {
            subject: subject,
            bodyHtml: bodyHtml,
            bodyText: bodyText,
            sender: sender,
            urls: urls,
        };

        // 4. --- API CALL TO LOCAL FASTAPI SERVICE (https://localhost:5000) ---
        try {
            const response = await fetch(API_URL, {
                method: "POST",
                headers: { 
                    "Content-Type": "application/json",
                    "X-Api-Key": API_KEY // Sending the API Key for authentication
                },
                body: JSON.stringify(payload),
            });

            if (!response.ok) {
                // Catches 401 (Invalid Key), 400 (Bad Request), etc.
                throw new Error(`HTTP error! Status: ${response.status} - ${response.statusText}`);
            }

            // The server responds with { "prediction": "phishing", "score": 0.95 }
            const data = await response.json();
            const prediction = data.prediction; 
            const score = (data.score * 100).toFixed(2);

            // 5. --- UI UPDATE ---
            if (prediction === "phishing") {
                resultBox.innerText = `⚠️ PHISHING DETECTED! (${score}% confidence)\nFrom: ${sender}\nSubject: ${subject}`;
                resultBox.className = "status-box red";
            } else if (prediction === "legitimate") {
                resultBox.innerText = `✅ Safe Email (${score}% confidence)\nFrom: ${sender}\nSubject: ${subject}`;
                resultBox.className = "status-box green";
            } else {
                 resultBox.innerText = "❌ Unknown prediction result from API.";
                 resultBox.className = "status-box red";
            }

        } catch (apiError) {
            console.error("API Fetch Error:", apiError);
            resultBox.innerText = `❌ Connection Error: Is the Python server running on ${API_URL}? Check terminal for errors.`;
            resultBox.className = "status-box red";
        }
        
      } catch (err) {
        console.error("General Add-in Error:", err);
        resultBox.innerText = "❌ General error running detection. Check browser console.";
        resultBox.className = "status-box red";
      }
    });
  }
});