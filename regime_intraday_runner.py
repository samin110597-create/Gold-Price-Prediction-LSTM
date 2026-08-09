import regime_intraday_model as rim


def safe_download(ticker, period, interval):
    periods = ('2y', '1y') if interval == '1h' else (period,)
    last_error = None
    for p in periods:
        try:
            df = rim.flatten(rim.yf.download(
                ticker,
                period=p,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            ))
            if not df.empty and 'Close' in df.columns:
                clean = df.dropna(subset=['Open', 'High', 'Low', 'Close']).copy()
                if not clean.empty:
                    return clean
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f'No usable data for {ticker} {interval}: {last_error}')


rim.download = safe_download

if __name__ == '__main__':
    rim.main()
