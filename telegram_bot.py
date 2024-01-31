import telegram
from telegram.ext import Updater, MessageHandler, Filters, CommandHandler
from intent_classifier import Classifier
from googletrans import Translator

BOT_TOKEN = ''

intent_classifier = Classifier()
translator = Translator()

bot = telegram.Bot(token=BOT_TOKEN)

def start(update, context):
    user_id = update.message.from_user.id
    context.bot.send_message(chat_id=user_id, text="Привет! Я здесь, чтобы ответить на твои вопросы по теме ритейла. Задавай, а я попробую догадаться, что ты хочешь.")

def handle_message(update, context):
    user_message = update.message.text
    intent = intent_classifier.predict(user_message)
    try:
        translated_intent = translator.translate(' '.join(intent[0].split('_')), src='en', dest='ru').text
        response_message = f"Ваше намерение, сударь: {translated_intent}"
    except:
        response_message = f"Your intent is {' '.join(intent[0].split('_'))}"
    update.message.reply_text(response_message)

def main():
    updater = Updater(token=BOT_TOKEN, use_context=True)
    dispatcher = updater.dispatcher

    start_handler = CommandHandler('start', start)

    dispatcher.add_handler(start_handler)

    message_handler = MessageHandler(Filters.text & ~Filters.command, handle_message)
    dispatcher.add_handler(message_handler)

    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
