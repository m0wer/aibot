# Ollama Telegram Bot

This project implements a Telegram bot powered by Ollama, capable of handling text and voice messages, with configurable language models and system prompts.

## Features

- Handles text and voice messages
- Uses Ollama for natural language processing
- Configurable Ollama URL and model
- Text-to-Speech (TTS) and Speech-to-Text (STT) capabilities
- Uses RQ (Redis Queue) for processing messages and audio conversion
- Different priority queues and a special GPU queue for Whisper STT using CUDA
- Allows users to adjust their system prompt
- Uses recent messages as context for responses
- Stores all interactions in SQLite using SQLModel
- Dockerized setup with configurable resource limits

## Prerequisites

- Docker and Docker Compose
- Ollama server running (can be on the same machine or a remote server)
- Telegram Bot Token (obtainable from BotFather on Telegram)


## Running the Bot

1. Build and start the services:
   ```
   cp .env.example .env
   # Edit the .env file and set your TELEGRAM_TOKEN and OLLAMA_URL
   docker compose up --build 
   ```

2. The bot should now be running and responding to messages on Telegram.

## Configuration

You can configure the following environment variables in your `.env` file:

- `TELEGRAM_TOKEN`: Your Telegram Bot Token (required)
- `OLLAMA_URL`: URL of your Ollama server (default: http://host.docker.internal:11434)
- `OLLAMA_MODEL`: Ollama model to use (default: llama3.1)

## Usage

- Start a chat with your bot on Telegram
- Send text messages or voice notes to interact with the bot
- Use the `/set_system_prompt` command to change the system prompt for the Ollama model

## Development

To modify the bot or add new features:

1. Edit the `bot.py` file to change the bot's behavior
1. Modify the `Dockerfile` if you need to add new system dependencies
1. Update the `requirements.in` file if you add or remove Python dependencies
1. Rebuild the `requirements.txt` file using `pip-compile requirements.in`
1. Rebuild and restart the services using `docker compose up --build`

## Troubleshooting

- If the bot can't connect to Ollama, make sure the `OLLAMA_URL` is correctly set and the Ollama server is running and accessible.
- For GPU support, ensure you have the necessary GPU drivers and Docker GPU runtime set up. Otherwise, disable the GPU reservations from the `docker compose.yaml` file with `sed -i 's/^.*devices:.*$/#&/' docker-compose.yaml`.
- Check the Docker logs for any error messages: `docker compose logs`.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
