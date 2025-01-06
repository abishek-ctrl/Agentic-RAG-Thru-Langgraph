const API_BASE_URL = "http://localhost:8000";

// Function to add a message to the chat window
function addMessageToChat(sender, message) {
    const chatWindow = document.getElementById("chatWindow");
    const messageDiv = document.createElement("div");
    messageDiv.classList.add("chat-message", sender);

    const messageContent = document.createElement("div");
    messageContent.classList.add("message");
    messageContent.textContent = message;

    messageDiv.appendChild(messageContent);
    chatWindow.appendChild(messageDiv);

    // Scroll to the bottom of the chat window
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

// Handle file upload
document.getElementById("uploadForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const files = document.getElementById("files").files;
    const formData = new FormData();

    for (let i = 0; i < files.length; i++) {
        formData.append("files", files[i]);
    }

    try {
        const response = await fetch(`${API_BASE_URL}/upload/`, {
            method: "POST",
            body: formData,
        });

        if (response.ok) {
            document.getElementById("uploadStatus").textContent = "Files uploaded successfully!";
        } else {
            document.getElementById("uploadStatus").textContent = "Failed to upload files.";
        }
    } catch (error) {
        document.getElementById("uploadStatus").textContent = "An error occurred while uploading files.";
    }
});

// Handle query submission
document.getElementById("queryForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = document.getElementById("query").value;

    // Add user query to the chat
    addMessageToChat("user", query);

    try {
        const response = await fetch(`${API_BASE_URL}/query/`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ query }),
        });

        if (response.ok) {
            const data = await response.json();
            addMessageToChat("bot", data.answer); // Add bot response to the chat
        } else {
            addMessageToChat("bot", "Failed to get an answer.");
        }
    } catch (error) {
        addMessageToChat("bot", "An error occurred while fetching the answer.");
    }

    // Clear the input field
    document.getElementById("query").value = "";
});