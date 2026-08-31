EXCHANGE_SUFFIXES = (
    ".NS", ".BO", ".L", ".IL", ".T", ".HK", ".TO", ".V", ".AX", ".NZ",
    ".DE", ".F", ".PA", ".AS", ".BR", ".MI", ".MC", ".SW", ".ST", ".CO",
    ".HE", ".OL", ".IC", ".KS", ".KQ", ".TW", ".SS", ".SZ", ".JK", ".KL",
    ".BK", ".SI", ".TA", ".SA", ".MX",
)

CURRENCY_SYMBOLS = {
    "INR": "₹",
    "USD": "$",
    "GBP": "£",
    "EUR": "€",
    "JPY": "¥",
    "HKD": "HK$",
    "AUD": "A$",
    "CAD": "C$",
    "CHF": "CHF ",
    "KRW": "₩",
    "TWD": "NT$",
    "CNY": "¥",
    "SGD": "S$",
    "BRL": "R$",
    "MXN": "MX$",
}


def normalize_ticker(raw: str) -> str:
    ticker = (raw or "").strip().upper().replace(" ", "")
    if not ticker:
        return ticker
    if ticker.startswith("^") or "=" in ticker:
        return ticker
    if any(ticker.endswith(suffix) for suffix in EXCHANGE_SUFFIXES):
        return ticker
    if "." in ticker:
        return ticker.replace(".", "-")
    return ticker
