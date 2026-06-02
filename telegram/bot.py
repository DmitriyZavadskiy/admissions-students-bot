import asyncio
import logging
import os
import re

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
)
from aiogram.utils.chat_action import ChatActionSender

from scripts.rag import RagLocalChat, sanitize_query


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("admissions_bot")

WELCOME = (
    "Привет! Я бот-помощник для абитуриентов НИУ ВШЭ (Москва).\n\n"
    "Выберите тему кнопкой ниже — я задам пару уточнений и дам точный ответ "
    "со ссылкой на источник. Или просто напишите вопрос текстом."
)

HELP = (
    "Я отвечаю по официальным материалам НИУ ВШЭ о поступлении.\n"
    "Нажмите тему ниже или напишите вопрос, например:\n"
    "• Сколько стоит Программная инженерия?\n"
    "• Когда приказы о зачислении?\n"
    "• Какие документы нужны для поступления?\n\n"
    "Если точного ответа в источниках нет, я честно об этом сообщу."
)

COST_PROMPT = (
    "Напишите название программы — отвечу по таблице стоимости. "
    "Например: «Сколько стоит Бизнес-информатика?»"
)

MAIN_KEYBOARD = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="💰 Стоимость"), KeyboardButton(text="📅 Сроки")],
        [KeyboardButton(text="📄 Документы"), KeyboardButton(text="🏠 Общежитие")],
        [KeyboardButton(text="📞 Контакты"), KeyboardButton(text="❓ Помощь")],
    ],
    resize_keyboard=True,
    is_persistent=True,
    input_field_placeholder="Спросите о поступлении…",
)

TOPIC_NODES = {
    "💰 Стоимость": "cost",
    "📅 Сроки": "dates",
    "📄 Документы": "docs",
    "🏠 Общежитие": "dorm",
    "📞 Контакты": "contacts",
}

MENU = {
    "cost": {
        "text": "Стоимость обучения. Выберите программу или группу:",
        "options": [
            ("Программы ФКН", "nav:cost_fcs"),
            ("Экономика", "q:cost_econ"),
            ("Другая программа — напишу сам", "type:cost"),
        ],
    },
    "cost_fcs": {
        "text": "Программа факультета компьютерных наук:",
        "options": [
            ("Прикладная математика и информатика", "q:cost_pmi"),
            ("Прикладной анализ данных", "q:cost_pad"),
            ("Компьютерные науки и анализ данных", "q:cost_knad"),
            ("Программная инженерия", "q:cost_pi"),
            ("Информатика и вычислительная техника", "q:cost_ivt"),
        ],
    },
    "dates": {
        "text": "Сроки приёмной кампании. Что интересует?",
        "options": [
            ("Подача документов", "q:dates_docs"),
            ("Публикация конкурсных списков", "q:dates_lists"),
            ("Приказы о зачислении", "q:dates_orders"),
            ("Вступительные испытания ВШЭ", "q:dates_exams"),
        ],
    },
    "docs": {
        "text": "Документы. Что интересует?",
        "options": [
            ("Перечень документов", "q:docs_list"),
            ("Как подать документы", "q:docs_how"),
            ("Документы для иностранцев", "q:docs_foreign"),
        ],
    },
    "dorm": {
        "text": "Общежитие. Что интересует?",
        "options": [
            ("Как заселиться", "q:dorm_how"),
            ("Как подать заявку", "q:dorm_apply"),
            ("Кому предоставляется", "q:dorm_who"),
        ],
    },
    "contacts": {
        "text": "Контакты. Что нужно?",
        "options": [
            ("Приёмная комиссия", "q:contacts_pk"),
            ("Электронная почта", "q:contacts_email"),
            ("Центр целевого обучения", "q:contacts_target"),
        ],
    },
}

QUERIES = {
    "cost_econ": "Сколько стоит обучение по программе Экономика в Москве?",
    "cost_pmi": "Сколько стоит обучение по программе Прикладная математика и информатика?",
    "cost_pad": "Сколько стоит обучение по программе Прикладной анализ данных?",
    "cost_knad": "Сколько стоит обучение по программе Компьютерные науки и анализ данных?",
    "cost_pi": "Сколько стоит обучение по программе Программная инженерия?",
    "cost_ivt": "Сколько стоит обучение по программе Информатика и вычислительная техника?",
    "dates_docs": "До какого числа подавать документы на поступление?",
    "dates_lists": "Когда публикуют конкурсные списки?",
    "dates_orders": "Когда издают приказы о зачислении на бюджетные места?",
    "dates_exams": "Когда проходят вступительные испытания НИУ ВШЭ?",
    "docs_list": "Какие документы нужны для поступления?",
    "docs_how": "Как подать документы на поступление?",
    "docs_foreign": "Какие документы нужны иностранным абитуриентам?",
    "dorm_how": "Как заселиться в общежитие?",
    "dorm_apply": "Как подать заявку на общежитие?",
    "dorm_who": "Кому предоставляется общежитие?",
    "contacts_pk": "Контакты приёмной комиссии",
    "contacts_email": "Электронная почта приёмной комиссии",
    "contacts_target": "Контакты центра целевого обучения",
}


FOLLOWUP_RE = re.compile(r"^(а|и|ну|тогда|ещё|еще)\b", re.IGNORECASE)
MODIFIERS = ("очн", "заочн", "онлайн", "англ", "бюджет", "платн", "за год", "в москв")


def is_followup(text: str) -> bool:
    low = text.lower()
    if FOLLOWUP_RE.match(low):
        return True
    if len(low.split()) <= 4 and any(modifier in low for modifier in MODIFIERS):
        return True
    return False


def menu_keyboard(node_id: str) -> InlineKeyboardMarkup:
    rows = [
        [InlineKeyboardButton(text=label, callback_data=data)]
        for label, data in MENU[node_id]["options"]
    ]
    return InlineKeyboardMarkup(inline_keyboard=rows)


def build_reply(result: dict) -> str:
    answer = (result.get("answer") or "").strip()
    if not answer:
        return "Не удалось сформировать ответ. Попробуйте переформулировать вопрос."

    links = []
    seen = set()
    for item in result.get("top_sources") or []:
        source = (item.get("source") or "").strip()
        if source.startswith("http") and source not in seen:
            seen.add(source)
            links.append(source)

    if links:
        answer += "\n\nПодробнее:\n" + "\n".join(f"• {link}" for link in links[:3])
    return answer


async def main() -> None:
    token = os.environ.get("BOT_TOKEN")
    if not token:
        raise SystemExit(
            "Не задан BOT_TOKEN. Получите токен у @BotFather и выполните: export BOT_TOKEN=<токен>"
        )

    use_llm = os.environ.get("RAG_USE_LLM", "1") not in {"0", "false", "False"}
    logger.info("Загрузка RAG-пайплайна (use_llm=%s)...", use_llm)
    rag = RagLocalChat(use_llm=use_llm)
    logger.info("RAG готов, чанков: %d", len(rag.chunks))

    bot = Bot(token=token)
    dp = Dispatcher()
    chat_subject: dict[int, str] = {}

    async def answer_question(chat_id: int, query: str, display: str, target: Message) -> None:
        async with ChatActionSender.typing(bot=bot, chat_id=chat_id):
            result = await asyncio.to_thread(rag.answer, query)
        await target.answer(
            f"❓ {display}\n\n{build_reply(result)}",
            reply_markup=MAIN_KEYBOARD,
            disable_web_page_preview=True,
        )

    @dp.message(CommandStart())
    async def on_start(message: Message) -> None:
        chat_subject.pop(message.chat.id, None)
        await message.answer(WELCOME, reply_markup=MAIN_KEYBOARD)

    @dp.message(Command("help"))
    async def on_help(message: Message) -> None:
        await message.answer(HELP, reply_markup=MAIN_KEYBOARD)

    async def replace_menu(callback: CallbackQuery, node_id: str) -> None:
        try:
            await callback.message.edit_text(MENU[node_id]["text"], reply_markup=menu_keyboard(node_id))
        except Exception:
            await callback.message.answer(MENU[node_id]["text"], reply_markup=menu_keyboard(node_id))

    async def drop_menu(callback: CallbackQuery) -> None:
        try:
            await callback.message.edit_reply_markup(reply_markup=None)
        except Exception:
            pass

    @dp.callback_query(F.data.startswith("nav:"))
    async def on_nav(callback: CallbackQuery) -> None:
        await replace_menu(callback, callback.data.split(":", 1)[1])
        await callback.answer()

    @dp.callback_query(F.data.startswith("type:"))
    async def on_type(callback: CallbackQuery) -> None:
        await drop_menu(callback)
        await callback.message.answer(COST_PROMPT, reply_markup=MAIN_KEYBOARD)
        await callback.answer()

    @dp.callback_query(F.data.startswith("q:"))
    async def on_leaf(callback: CallbackQuery) -> None:
        question = QUERIES[callback.data.split(":", 1)[1]]
        chat_subject[callback.message.chat.id] = question
        await callback.answer()
        await drop_menu(callback)
        await answer_question(callback.message.chat.id, question, question, callback.message)

    @dp.message(F.text)
    async def on_text(message: Message) -> None:
        text = sanitize_query(message.text or "")
        if not text:
            return
        if text == "❓ Помощь":
            await message.answer(HELP, reply_markup=MAIN_KEYBOARD)
            return
        if text in TOPIC_NODES:
            node_id = TOPIC_NODES[text]
            await message.answer(MENU[node_id]["text"], reply_markup=menu_keyboard(node_id))
            return

        chat_id = message.chat.id
        if is_followup(text) and chat_id in chat_subject:
            query = f"{chat_subject[chat_id]} {text}"
        else:
            chat_subject[chat_id] = text
            query = text
        await answer_question(chat_id, query, text, message)

    @dp.message()
    async def on_other(message: Message) -> None:
        await message.answer("Пожалуйста, пришлите вопрос текстом.", reply_markup=MAIN_KEYBOARD)

    logger.info("Бот запущен (long polling). Остановка: Ctrl+C")
    await dp.start_polling(bot)
