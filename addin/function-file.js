// function-file.js (Base template)

// Define the function that is referenced in your manifest.xml
function phishingCheck(event) {
    // This is where you would put logic to run silently in the background (e.g., OnMessageRead event)

    // *** CRITICAL ***: Always call event.completed() when your function is done.
    event.completed(); 
}

// Associate the function name with the actual function implementation
// This is how Office.js knows which function to call when an event triggers.
Office.actions.associate("phishingCheck", phishingCheck);