document.addEventListener("DOMContentLoaded", () => {
  const chatUI = {
    userInput: document.getElementById("userInput"),
    sendButton: document.getElementById("sendMessageBtn"),
    chatMessages: document.getElementById("chatMessages"),

    init: function () {
      this.sendButton.addEventListener("click", (e) => {
        e.preventDefault();
        this.handleSend();
      });

      this.userInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          this.handleSend();
        }
      });
    },

    handleSend: async function () {
      const message = this.userInput.value.trim();
      if (!message) return;

      this.addMessage(message, "user");
      this.userInput.value = "";
      this.showLoading();

      try {
       // In chat.js
          // Update the fetch call to use JSON format
          const response = await fetch('/api/generate', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json', // Changed from x-www-form-urlencoded
            },
            body: JSON.stringify({
              prompt: message
            })
          });

        if (!response.ok) throw new Error('API request failed');

        const data = await response.json();
        this.addMessage(data.generated_text, "ai");
      } catch (error) {
        console.error("Error:", error);
        this.addMessage("Something went wrong. Please try again.", "error");
      } finally {
        this.hideLoading();
      }

      this.scrollToBottom();
    },

    addMessage: function (text, sender) {
      const messageDiv = document.createElement("div");
      messageDiv.className = `${sender}-message message`;
      messageDiv.innerHTML = `
        <div class="message-content">${this.sanitizeText(text)}</div>
        <div class="message-timestamp">${new Date().toLocaleTimeString([], { 
          hour: '2-digit', 
          minute: '2-digit' 
        })}</div>
      `;
      this.chatMessages.appendChild(messageDiv);
    },

    sanitizeText: function (text) {
      const div = document.createElement('div');
      div.textContent = text;
      return div.innerHTML.replace(/\n/g, '<br>');
    },

    scrollToBottom: function () {
      this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    },

    showLoading: function () {
      const loadingDiv = document.createElement("div");
      loadingDiv.className = "ai-message message loading";
      loadingDiv.innerHTML = `
        <div class="message-content">
          <div class="typing-indicator">
            <div class="dot"></div>
            <div class="dot"></div>
            <div class="dot"></div>
          </div>
        </div>
      `;
      this.chatMessages.appendChild(loadingDiv);
      this.scrollToBottom();
    },

    hideLoading: function () {
      const loadingElements = this.chatMessages.querySelectorAll('.loading');
      loadingElements.forEach(el => el.remove());
    }
  };

  chatUI.init();
});