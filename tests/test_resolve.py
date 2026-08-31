from nlp_brief import resolve_ticker_local


def test_resolves_company_name_not_title_case_token():
    assert resolve_ticker_local("Apple") == "AAPL"
    assert resolve_ticker_local("outlook for Microsoft") == "MSFT"


def test_resolves_plain_ticker():
    assert resolve_ticker_local("RELIANCE.NS") == "RELIANCE.NS"
    assert resolve_ticker_local("aapl") == "AAPL"
