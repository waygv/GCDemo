# 📦 Required Libraries
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import gradio as gr
from gradio import Blocks
from groq import Groq
import os
from datetime import datetime
from markdown import markdown
import json
from serpapi.google_search import GoogleSearch
import wikipedia
import requests
from bs4 import BeautifulSoup

# --- API Keys (⚠️ Hardcoded) ---
GROQ_API_KEY = "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"
SERP_API_KEY = "2a7eb96d70216cfaf82bb38cb797af2d363fa989f7c00f49bf9e2a6c0c4c80a2"
client = Groq(api_key=GROQ_API_KEY)

# --- Fetch Company Financial Data ---
def fetch_company_data(ticker):
    try:
        ticker = ticker.upper()
        folder = f"company_data/{ticker}"
        os.makedirs(folder, exist_ok=True)

        stock = yf.Ticker(ticker)
        hist = stock.history(period="2y")

        income = stock.financials
        balance = stock.balance_sheet
        cashflow = stock.cashflow
        esg = stock.sustainability if stock.sustainability is not None else pd.DataFrame()

        income.to_csv(f"{folder}/income_statement.csv")
        balance.to_csv(f"{folder}/balance_sheet.csv")
        cashflow.to_csv(f"{folder}/cashflow.csv")
        hist.to_csv(f"{folder}/historical_data.csv")
        if not esg.empty:
            esg.to_csv(f"{folder}/esg.csv")

        try:
            news = stock.news[:5]
            news_txt = '\n'.join([f"{item['title']} ({item['publisher']})" for item in news])
        except:
            news_txt = "News not available."
        with open(f"{folder}/news_headlines.txt", "w", encoding="utf-8") as f:
            f.write(news_txt)

        if len(hist) >= 200:
            hist["SMA_200"] = hist["Close"].rolling(window=200).mean()
            fig1 = px.line(hist, x=hist.index, y="SMA_200", title=f"{ticker} – 200-Day SMA")
        else:
            fig1 = go.Figure().add_annotation(text="Insufficient data for SMA 200")

        fig2 = px.bar(hist, x=hist.index, y="Volume", title=f"{ticker} – Volume")

        fig3 = go.Figure(data=[go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close']
        )])
        fig3.update_layout(title=f"{ticker} – Candlestick", xaxis_title="Date", yaxis_title="Price")

        return (
            income.head().T.to_html(classes="table table-striped", border=0),
            balance.head().T.to_html(classes="table table-striped", border=0),
            cashflow.head().T.to_html(classes="table table-striped", border=0),
            esg.head().reset_index().to_html(classes="table table-striped", border=0)
                if not esg.empty else "<b>No ESG data available.</b>",
            news_txt,
            fig1, fig2, fig3,
            folder,
            ticker
        )

    except Exception as e:
        return (
            "<b>Error loading income statement</b>",
            "<b>Error loading balance sheet</b>",
            "<b>Error loading cash flow</b>",
            "<b>Error loading ESG</b>",
            "Error loading news",
            go.Figure().add_annotation(text="Chart Error"),
            go.Figure().add_annotation(text="Chart Error"),
            go.Figure().add_annotation(text="Chart Error"),
            "",
            ""
        )

# --- Generate AI Investment Report ---
def generate_llama_report(folder, ticker):
    def try_read(path, read_csv=False):
        try:
            if read_csv:
                return pd.read_csv(path).head().to_string(index=False)
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except:
            return "Not available."

    data = {
        "Income": try_read(f"{folder}/income_statement.csv", read_csv=True),
        "Balance": try_read(f"{folder}/balance_sheet.csv", read_csv=True),
        "Cashflow": try_read(f"{folder}/cashflow.csv", read_csv=True),
        "ESG": try_read(f"{folder}/esg.csv", read_csv=True),
        "News": try_read(f"{folder}/news_headlines.txt"),
        "History": try_read(f"{folder}/historical_data.csv", read_csv=True),
    }

    prompt = f"""
You are a professional equity analyst.
Write a deep-dive investment report for {ticker} based on:

Income:\n{data['Income']}
Balance:\n{data['Balance']}
Cashflow:\n{data['Cashflow']}
ESG:\n{data['ESG']}
News:\n{data['News']}
History:\n{data['History']}
...
"""

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=4000
    )

    report = response.choices[0].message.content
    with open(f"{folder}/investment_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    return markdown(report, extensions=["tables"])

# --- Other helper functions (get_company_name, patents, subsidiaries) ---
# [Your existing functions go here unchanged]

# --- Full Combined Workflow ---
def full_analysis(ticker):
    income, balance, cashflow, esg, news, fig1, fig2, fig3, folder, ticker = fetch_company_data(ticker)
    report = generate_llama_report(folder, ticker)

    try:
        cname = get_company_name(ticker)
        patents_raw = search_patents(cname)
        patents = []
        for p in patents_raw:
            analysis = analyze_patent(p["title"], p.get("snippet", ""))
            patents.append({
                "title": p["title"],
                "link": p.get("link"),
                "analysis": analysis
            })
        if not patents:
            patents = [{"info": "No patents found or mismatch with company name."}]
    except Exception as e:
        patents = [{"error": f"Patent analysis failed: {str(e)}"}]

    patent_table = format_patents_to_html(patents)
    subsidiary_summary = get_subsidiary_summary(ticker)

    return (
        income, balance, cashflow, esg, news,
        fig1, fig2, fig3, report, patent_table, subsidiary_summary
    )

# --- Gradio App ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AI-Powered Company + Patent + Subsidiary Analyzer")

    with gr.Row():
        ticker_input = gr.Textbox(label="Enter Stock Ticker (e.g. AAPL, MSFT, INFY.NS)")
        analyze_btn = gr.Button("Run Full Analysis")

    with gr.Tabs():
        with gr.Tab("Financial Analysis"):
            with gr.Row():
                income_html = gr.HTML(label="Income Statement")
                balance_html = gr.HTML(label="Balance Sheet")
            with gr.Row():
                cashflow_html = gr.HTML(label="Cash Flow")
                esg_html = gr.HTML(label="ESG Metrics")
            news_txt = gr.Textbox(label="Latest News", lines=4)

            with gr.Row():
                chart1 = gr.Plot(label="200-Day SMA")
                chart2 = gr.Plot(label="Volume")
            chart3 = gr.Plot(label="Candlestick")

            report_html = gr.HTML(label="AI-Generated Investment Report")

        with gr.Tab("Patent Summary"):
            patent_html = gr.HTML(label="Patent Analysis")

        with gr.Tab("Subsidiary Summary"):
            subsidiary_html = gr.HTML(label="Subsidiary Analysis")

    analyze_btn.click(
        fn=full_analysis,
        inputs=ticker_input,
        outputs=[
            income_html, balance_html, cashflow_html, esg_html,
            news_txt, chart1, chart2, chart3,
            report_html, patent_html, subsidiary_html
        ]
    )

# --- Entry Point for Render ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
