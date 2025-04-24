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
        if (e.key === "Enter") {
          e.preventDefault();
          this.handleSend();
        }
      });
    },

    handleSend: function () {
      const message = this.userInput.value.trim();
      if (message) {
        this.addMessage(message, "user");
        this.simulateAIResponse();
        this.userInput.value = "";
        this.scrollToBottom();
      }
    },

    addMessage: function (text, sender) {
      const messageDiv = document.createElement("div");
      messageDiv.className = `${sender}-message message`;
      messageDiv.textContent = text;
      this.chatMessages.appendChild(messageDiv);
    },

    simulateAIResponse: function () {
      setTimeout(() => {
        this.addMessage("Here's a sample AI response...", "ai");
        this.scrollToBottom();
      }, 1000);
    },

    scrollToBottom: function () {
      this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    },
  };

  chatUI.init();
});
