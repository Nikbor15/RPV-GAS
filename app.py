# app.py
# -*- coding: utf-8 -*-
import streamlit as st
from __future__ import annotations

"""
Leitura Integrada IBOV — enxuta e integrada (ATUALIZADA)

• GEX: CALL/PUT GEX por strike (barras) + NET (linha). Call/Put Wall e Gamma Flip.
• IV: usa “Vol Impl” quando existir; sem inferir pelo mid (apenas normalização).
• GEX: usa gamma da planilha; Black–Scholes como fallback para NaN/zeros.
• Vencimento/DTE: contado até a próxima 3ª sexta-feira (OpEx BR).
• Amostragem: tick de strike inferido automaticamente.
• Streamlit: UI multi-ativos (Súmula, Painéis, GEX, Tabelas, ML).
"""

# ==========================
# Dependências
# ==========================
import os, io, glob, math, sys, warnings, time, re, unicodedata
import numpy as np, pandas as pd
from itertools import combinations

# Plot
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pandas.tseries.offsets import BDay

# Econometria / ML
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.arima.model import ARIMA#
from statsmodels.tsa.statespace.sarimax import SARIMAX


try:
    from arch import arch_model
    _HAS_ARCH = True
except Exception:
    arch_model = None
    _HAS_ARCH = False

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score


from math import erf, sqrt, log, exp

# pmdarima — opcional
try:
    from pmdarima import auto_arima as _auto_arima
except Exception:
    _auto_arima = None

# TensorFlow é opcional; se não existir, DL fica desativado
try:
    import tensorflow as tf  # noqa
    from tensorflow import keras  # noqa
    _HAS_TF = True
except Exception:
    _HAS_TF = False

try:
    from statsmodels.tools.sm_exceptions import ValueWarning as SMValueWarning
except Exception:
    class SMValueWarning(Warning): ...

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # reduz verbosidade do TF


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

warnings.filterwarnings("ignore", category=SMValueWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice*")

# ==========================
# Config / Estilo
# ==========================
CONTRACT_MULTIPLIER = 100.0  # B3: 100 ações por contrato
DEC_COMMA = True

JP_COLORS = {
    "bg": "#0B1E2D", "panel": "#0F2434", "grid": "rgba(255,255,255,0.06)",
    "txt": "#E7EEF3", "accent": "#18A999", "gold": "#C8A35B", "red": "#D66A6A",
    "green":"#53B67E", "blue": "#86A6C9", "grey": "#7F8C8D"
}

def _apply_plotly_theme(fig: go.Figure, title: str | None = None):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=JP_COLORS["bg"],
        plot_bgcolor=JP_COLORS["panel"],
        font=dict(color=JP_COLORS["txt"]),
        title=title or fig.layout.title.text,
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)")
    )
    fig.update_xaxes(showgrid=True, gridcolor=JP_COLORS["grid"])
    fig.update_yaxes(showgrid=True, gridcolor=JP_COLORS["grid"])
    return fig

# ==========================
# Utils
# ==========================
def to_float(x):
    import re, numpy as np, pandas as pd
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float, np.floating)): return float(x)

    s = str(x).strip()
    if s in {"", "-", "None", "nan"}: return np.nan

    # sufixos
    suf = s[-1:].lower()
    mult = 1.0
    if suf in {"%", "k", "m"}:
        s = s[:-1]
        mult = {"%": 0.01, "k": 1e3, "m": 1e6}[suf]

    # limpar lixo/moeda
    s = s.replace("\xa0", "").replace("\u00a0", "")
    s = re.sub(r"(?i)(r\$|us\$|brl|usd|\$)", "", s)
    s = re.sub(r"[^\d.,\-+]", "", s)

    # escolher separador decimal dinamicamente
    if s.count(",") and s.count("."):
        # usa o separador mais à direita como decimal
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif s.count(",") == 1 and s.count(".") == 0:
        s = s.replace(",", ".")
    else:
        s = s.replace(",", "")  # só pontos ou nenhum → mantém ponto como decimal

    try:
        val = float(s)
    except Exception:
        return np.nan
    return val * mult

def mid_from(bid, ask):
    b, a = to_float(bid), to_float(ask)
    if np.isnan(b) and np.isnan(a): return np.nan
    if np.isnan(b): return a
    if np.isnan(a): return b
    return (a+b)/2.0

def _runs_of_equals(series):
    if len(series)==0: return []
    idx, vals, runs, start = series.index, series.values, [], 0
    for i in range(1, len(vals)):
        if vals[i]!=vals[i-1]:
            runs.append((idx[start], idx[i-1], vals[i-1], i-1-start+1)); start=i
    runs.append((idx[start], idx[len(vals)-1], vals[len(vals)-1], len(vals)-start))
    return runs

def safe_median(series_like):
    try:
        s = pd.Series(series_like, dtype="float64").replace([np.inf,-np.inf], np.nan).dropna()
        return float(s.median()) if not s.empty else np.nan
    except Exception:
        return np.nan

def _top_features(coef_dict: dict, k: int = 5):
    if not coef_dict: return [], []
    s = pd.Series(coef_dict, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty: return [], []
    pos = s.sort_values(ascending=False).head(k)
    neg = s.sort_values(ascending=True).head(k)
    return list(pos.items()), list(neg.items())

def _pick_last_options_day(df: pd.DataFrame, min_strikes: int = 5):
    if "date" not in df.columns:
        return None
    t = df.copy()
    t["date_n"]   = pd.to_datetime(t["date"], errors="coerce").dt.normalize()
    t["strike_n"] = pd.to_numeric(t.get("strike"), errors="coerce")
    t["oi_n"]     = pd.to_numeric(t.get("oi"),     errors="coerce")
    t["gamma_n"]  = pd.to_numeric(t.get("gamma"),  errors="coerce")
    t["iv_n"]     = pd.to_numeric(t.get("iv"),     errors="coerce")
    t["mid_n"]    = pd.to_numeric(t.get("mid"),    errors="coerce")

    m = (t["strike_n"] > 0) & (
        (t["oi_n"] > 0) | (t["gamma_n"].fillna(0) != 0) | (t["iv_n"] > 0) | (t["mid_n"] > 0)
    )
    if not m.any():
        return None

    cnt = t.loc[m].groupby("date_n")["strike_n"].nunique().sort_index()
    if cnt.empty:                         # <- NOVO: nada agrupado (datas viraram NaT)
        return None

    cand = cnt[cnt >= int(min_strikes)]
    return (cand.index[-1] if not cand.empty else cnt.index[-1])

def _norm_col(s: str) -> str:
    s = str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    s = re.sub(r"[._\-/%()]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\.\d+$", "", s)  # pandas suffix (.1, .2)
    return s

def _find_contains(cols, tokens):
    nmap = {c: _norm_col(c) for c in cols}
    for c, n in nmap.items():
        if any(tok in n for tok in tokens):
            return c
    return None

def _find_col_fuzzy(cols, targets):
    nmap = {c: _norm_col(c) for c in cols}
    tnorm = {_norm_col(t) for t in targets}
    for c, n in nmap.items():
        if n in tnorm: return c
    for c, n in nmap.items():
        if any(n.startswith(t) or t in n for t in tnorm): return c
    raise KeyError(f"Não encontrei nenhuma das colunas {targets} em: {list(cols)}")

def _safe_find(cols, targets, contains_tokens):
    try:
        return _find_col_fuzzy(cols, targets)
    except KeyError:
        return _find_contains(cols, contains_tokens)

# ==========================
# Leitura de dados — robusta
# ==========================
def _smart_read_excel_table(xlsx_path_or_bytes, sheet_name=0):
    """
    Lê a planilha detectando a linha de cabeçalho (mesmo quando a 1ª linha é CALL/STRIKE/PUT).
    Não sai apagando 'Unnamed'; remove apenas colunas 100% vazias.
    Mantém tudo como string; números são convertidos depois.
    """
    import pandas as pd

    raw = pd.read_excel(
        xlsx_path_or_bytes, sheet_name=sheet_name, engine="openpyxl",
        header=None, dtype=str
    )

    tokens = [t.lower() for t in [
        "strike","exerc","preço","preco","preco de exerc","preço de exerc",
        "c. abertos","contratos em aberto","open interest","oi",
        "gamma","vol","iv","bid","ask","delta","vega","theta","tipo","call","put"
    ]]

    header_row = None
    scan_rows = min(200, len(raw))  # ↑ ampliado
    for i in range(scan_rows):
        row = [str(x).lower() for x in raw.iloc[i].fillna("").astype(str).tolist()]
        if any(any(tok in cell for tok in tokens) for cell in row):
            header_row = i
            break

    if header_row is not None:
        hdr = []
        for j, x in enumerate(raw.iloc[header_row].tolist()):
            sx = str(x).strip()
            hdr.append(sx if sx not in {"", "nan", "None"} else f"col_{j}")
        df = raw.iloc[header_row+1:].copy()
        df.columns = hdr
    else:
        df = pd.read_excel(
            xlsx_path_or_bytes, sheet_name=sheet_name, engine="openpyxl",
            dtype=str
        )

    # remova apenas colunas completamente vazias
    non_empty = df.apply(lambda c: c.astype(str).str.strip().replace({"nan": "", "None": ""}).ne("").any())
    df = df.loc[:, non_empty.values].copy()
    return df


def _read_chain_longform(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Fallback para planilhas em FORMATO LONGO:
    colunas do tipo [Tipo/CallPut], Strike/Exercício, OI, Gamma, IV, Bid, Ask, Date...
    """
    cols = list(raw.columns)

    def pick(targets, contains):
        c = _safe_find(cols, targets, contains)
        return c

    col_type   = pick(["type","tipo","callput","opcao","option type","c/p"], ["call","put","tipo","c/p","cp"])
    col_strike = pick(
        ["strike","preço de exercicio","preco de exercicio","exercicio","exerc"],
        ["strike","exerc"]
    )
    col_oi     = pick(["c. abertos","abertos","open interest","oi"], ["abert","interest","oi"])
    col_gamma  = pick(["gamma"], ["gamma","gama"])
    col_iv     = pick(["vol impl","iv","volatilidade implicita","volatilidade implied"], ["vol","iv"])
    col_bid    = pick(["bid","compra","preço compra","preco compra"], ["bid","compra"])
    col_ask    = pick(["ask","venda","preço venda","preco venda","offer","ofertas"], ["ask","venda","ofe"])
    col_date   = pick(["data","date","dt","timestamp"], ["data","date"])

    def s(col):
        return raw[col] if (col is not None and col in raw.columns) else pd.Series(np.nan, index=raw.index)

    # normalizações
    tp = s(col_type).astype(str).str.upper().str.strip().str[0]   # 'C' ou 'P' do primeiro caractere
    tp = tp.replace({"C":"C","P":"P"}).where(tp.isin(["C","P"]), np.nan)

    out = pd.DataFrame({
        "date":   pd.to_datetime(s(col_date), dayfirst=True, errors="coerce"),
        "type":   tp,
        "strike": pd.to_numeric(s(col_strike).apply(to_float), errors="coerce"),
        "oi":     pd.to_numeric(s(col_oi).apply(to_float), errors="coerce"),
        "gamma":  pd.to_numeric(s(col_gamma).apply(to_float), errors="coerce"),
        "iv":     _normalize_iv_series(s(col_iv)),
        "bid":    pd.to_numeric(s(col_bid).apply(to_float), errors="coerce"),
        "ask":    pd.to_numeric(s(col_ask).apply(to_float), errors="coerce"),
    })

    out = out.dropna(subset=["strike","type"], how="any")
    return out


def _read_chain_and_daily_matrixlike(xlsx_path_or_bytes, sheet_name=0):
    """
    1) tenta layout MATRIZ (CALL | STRIKE | PUT)
    2) se não achar STRIKE, tenta layout LONGO (fallback)
    3) retorna (chain, daily)
    """
    raw = _smart_read_excel_table(xlsx_path_or_bytes, sheet_name=sheet_name)
    raw.columns = [str(c).strip() for c in raw.columns]

    def _norm(s: str) -> str:
        s = unicodedata.normalize("NFKD", str(s)).encode("ascii","ignore").decode("ascii")
        s = re.sub(r"[._\-/%()]+", " ", s.lower()).strip()
        s = re.sub(r"\s+", " ", s)
        return s

    cols = list(raw.columns)
    strike_idx = next((i for i, c in enumerate(cols)
                       if ("strike" in _norm(c)) or ("exerc" in _norm(c))), None)

    # ---------- fallback: LONGO ----------
    if strike_idx is None:
        try:
            chain_long = _read_chain_longform(raw)
            daily = _read_daily_only(raw)
            if isinstance(chain_long, pd.DataFrame) and not chain_long.empty:
                return chain_long, daily
        except Exception:
            pass
        # nem matriz nem longo → só diário
        return pd.DataFrame(columns=["type","strike","oi","iv","gamma","bid","ask","date"]), _read_daily_only(raw)

    # ---------- MATRIZ (original) ----------
    L = raw.iloc[:, :strike_idx]
    R = raw.iloc[:, strike_idx+1:]
    strike = raw.iloc[:, strike_idx].apply(to_float)

    def _pick_col(df_side, prefer, contains):
        try:
            col = _find_col_fuzzy(df_side.columns, prefer)
        except Exception:
            col = _find_contains(df_side.columns, contains)
        return col

    def _series_or_nan(df_side, col):
        return df_side[col].apply(to_float) if (col is not None and col in df_side.columns) \
               else pd.Series(np.nan, index=raw.index)

    # CALLs
    col_oi_L    = _pick_col(L, ["C. Abertos","Abertos","Open Interest","OI"], ["abert","interest","oi"])
    col_gamma_L = _pick_col(L, ["Gamma"], ["gamma","gama"])
    col_iv_L    = _pick_col(L, ["Vol Impl","IV","Volatilidade Implicita","Volatilidade Implied"], ["vol","iv"])
    col_bid_L   = _pick_col(L, ["Bid","Compra","Preço Compra","Preco Compra"], ["bid","compra"])
    col_ask_L   = _pick_col(L, ["Ask","Venda","Preço Venda","Preco Venda","Offer","Ofertas"], ["ask","venda","ofe"])

    C = pd.DataFrame({
        "date":   pd.NaT,
        "type":   "C",
        "strike": pd.to_numeric(strike, errors="coerce"),
        "oi":     _series_or_nan(L, col_oi_L),
        "gamma":  _series_or_nan(L, col_gamma_L),
        "iv":     _normalize_iv_series(_series_or_nan(L, col_iv_L)),
        "bid":    _series_or_nan(L, col_bid_L),
        "ask":    _series_or_nan(L, col_ask_L),
    })

    # PUTs
    col_oi_R    = _pick_col(R, ["C. Abertos","Abertos","Open Interest","OI"], ["abert","interest","oi"])
    col_gamma_R = _pick_col(R, ["Gamma"], ["gamma","gama"])
    col_iv_R    = _pick_col(R, ["Vol Impl","IV","Volatilidade Implicita","Volatilidade Implied"], ["vol","iv"])
    col_bid_R   = _pick_col(R, ["Bid","Compra","Preço Compra","Preco Compra"], ["bid","compra"])
    col_ask_R   = _pick_col(R, ["Ask","Venda","Preço Venda","Preco Venda","Offer","Ofertas"], ["ask","venda","ofe"])

    P = pd.DataFrame({
        "date":   pd.NaT,
        "type":   "P",
        "strike": pd.to_numeric(strike, errors="coerce"),
        "oi":     _series_or_nan(R, col_oi_R),
        "gamma":  _series_or_nan(R, col_gamma_R),
        "iv":     _normalize_iv_series(_series_or_nan(R, col_iv_R)),
        "bid":    _series_or_nan(R, col_bid_R),
        "ask":    _series_or_nan(R, col_ask_R),
    })

    chain = pd.concat([C, P], ignore_index=True).dropna(subset=["strike"])
    daily = _read_daily_only(raw)
    return chain, daily

def _read_daily_only(df: pd.DataFrame):
    cols = {c.lower().strip(): c for c in df.columns}
    def pick(*names):
        for nm in names:
            if nm.lower() in cols: return cols[nm.lower()]
        return None

    date_raw = df.get(pick("Data","Date"))
    date = pd.to_datetime(date_raw, errors="coerce", dayfirst=False)
    if date.isna().all():
    # tenta o formato BR se a primeira passada falhar
        date = pd.to_datetime(date_raw, errors="coerce", dayfirst=True)

    open_ = df.get(pick("Abertura","Open"))
    high  = df.get(pick("Máxima","Maxima","High"))
    low   = df.get(pick("Mínima","Minima","Low"))
    close = df.get(pick("Fechamento","Close","Último","Ultimo"))
    vol   = df.get(pick("Volume Quantidade","Volume"))
    aggr  = df.get(pick("Agressão","Agressao","TR - Volume de Agressão - Saldo"))

    open_ = pd.Series(open_).apply(to_float) if open_ is not None else np.nan
    high  = pd.Series(high ).apply(to_float) if high  is not None else np.nan
    low   = pd.Series(low  ).apply(to_float) if low   is not None else np.nan
    close = pd.Series(close).apply(to_float) if close is not None else np.nan
    vol   = pd.Series(vol  ).apply(to_float) if vol   is not None else np.nan
    aggr  = pd.Series(aggr ).apply(to_float) if aggr  is not None else np.nan

    daily = (pd.DataFrame({"date": date, "open": open_, "high": high, "low": low, "close": close, "vol": vol, "aggr": aggr})
               .dropna(subset=["date","close"])
               .sort_values("date").set_index("date"))
    tr = (daily["high"] - daily["low"]).fillna(0.0)
    daily["atr14"] = tr.rolling(14, min_periods=1).mean()
    daily["ma200"] = daily["close"].rolling(200, min_periods=1).mean()
    return daily

def _read_chain_and_daily_matrixlike(xlsx_path_or_bytes, sheet_name=0):
    raw = _smart_read_excel_table(xlsx_path_or_bytes, sheet_name=sheet_name)
    raw.columns = [str(c).strip() for c in raw.columns]

    def _norm(s: str) -> str:
        s = unicodedata.normalize("NFKD", str(s)).encode("ascii","ignore").decode("ascii")
        s = re.sub(r"[._\-/%()]+", " ", s.lower()).strip()
        s = re.sub(r"\s+", " ", s)
        return s

    cols = list(raw.columns)
    strike_idx = next((i for i,c in enumerate(cols)
                       if ("strike" in _norm(c)) or ("exerc" in _norm(c))), None)

    # Fallback: inferir por padrão numérico se não achou por nome
    if strike_idx is None:
        cand_idx, cand_score = None, -1
        for i, c in enumerate(cols):
            v = pd.Series(raw.iloc[:, i]).apply(to_float)
            pos = v[(v > 0) & np.isfinite(v)]
            score = pos.nunique()
            if score >= 5:
                inc = (pos.diff().dropna() >= 0).mean() if len(pos) > 6 else 0
                score += 2*inc
            if score > cand_score:
                cand_idx, cand_score = i, score
        if cand_score >= 5:
            strike_idx = cand_idx

    if strike_idx is None:
        # sem strike → só diário
        return pd.DataFrame(columns=["type","strike","oi","iv","gamma","bid","ask","date"]), _read_daily_only(raw)

    # usa TODAS as colunas à esquerda/direita do strike
    L = raw.iloc[:, :strike_idx]
    R = raw.iloc[:, strike_idx+1:]
    strike = raw.iloc[:, strike_idx].apply(to_float)

    def _pick_col(df_side, prefer, contains):
        try:
            col = _find_col_fuzzy(df_side.columns, prefer)
        except Exception:
            col = _find_contains(df_side.columns, contains)
        return col

    def _series_or_nan(df_side, col):
        return df_side[col].apply(to_float) if (col is not None and col in df_side.columns) \
               else pd.Series(np.nan, index=raw.index)

    # CALLs (lado esquerdo)
    col_oi_L    = _pick_col(L, ["C. Abertos","Abertos","Open Interest","OI"], ["abert","interest","oi"])
    col_gamma_L = _pick_col(L, ["Gamma"], ["gamma","gama"])
    col_iv_L    = _pick_col(L, ["Vol Impl","IV","Volatilidade Implicita","Volatilidade Implied"], ["vol","iv"])
    col_bid_L   = _pick_col(L, ["Bid","Compra","Pre\u00e7o Compra","Preco Compra"], ["bid","compra"])
    col_ask_L   = _pick_col(L, ["Ask","Venda","Pre\u00e7o Venda","Preco Venda","Offer","Ofertas"], ["ask","venda","ofe"])

    C = pd.DataFrame({
        "date":   pd.NaT,
        "type":   "C",
        "strike": pd.to_numeric(strike, errors="coerce"),
        "oi":     _series_or_nan(L, col_oi_L),
        "gamma":  _series_or_nan(L, col_gamma_L),
        "iv":     _normalize_iv_series(_series_or_nan(L, col_iv_L)),
        "bid":    _series_or_nan(L, col_bid_L),
        "ask":    _series_or_nan(L, col_ask_L),
    })

    # PUTs (lado direito)
    col_oi_R    = _pick_col(R, ["C. Abertos","Abertos","Open Interest","OI"], ["abert","interest","oi"])
    col_gamma_R = _pick_col(R, ["Gamma"], ["gamma","gama"])
    col_iv_R    = _pick_col(R, ["Vol Impl","IV","Volatilidade Implicita","Volatilidade Implied"], ["vol","iv"])
    col_bid_R   = _pick_col(R, ["Bid","Compra","Pre\u00e7o Compra","Preco Compra"], ["bid","compra"])
    col_ask_R   = _pick_col(R, ["Ask","Venda","Pre\u00e7o Venda","Preco Venda","Offer","Ofertas"], ["ask","venda","ofe"])

    P = pd.DataFrame({
        "date":   pd.NaT,
        "type":   "P",
        "strike": pd.to_numeric(strike, errors="coerce"),
        "oi":     _series_or_nan(R, col_oi_R),
        "gamma":  _series_or_nan(R, col_gamma_R),
        "iv":     _normalize_iv_series(_series_or_nan(R, col_iv_R)),
        "bid":    _series_or_nan(R, col_bid_R),
        "ask":    _series_or_nan(R, col_ask_R),
    })

    chain = pd.concat([C, P], ignore_index=True).dropna(subset=["strike"])
    daily = _read_daily_only(raw)
    return chain, daily

def read_chain_and_daily_auto(xlsx_path_or_bytes, sheet_name=0):
    """
    Varre as abas quando a cadeia vier vazia da aba pedida.
    Escolhe a aba com MAIOR número de strikes válidos (únicos > 0).
    Para o diário, fica com a aba que tiver mais linhas de OHLC.
    Sempre retorna (chain, daily), mesmo em cenários ruins.
    """
    # 1) tenta a aba pedida
    try:
        chain, daily = _read_chain_and_daily_matrixlike(xlsx_path_or_bytes, sheet_name=sheet_name)
    except Exception:
        chain, daily = pd.DataFrame(), pd.DataFrame()

    def _score_chain(ch):
        if not isinstance(ch, pd.DataFrame) or ch.empty:
            return 0
        s = pd.to_numeric(ch.get("strike"), errors="coerce")
        return int(s[s > 0].nunique())   # strikes únicos > 0

    if _score_chain(chain) >= 2:
        return chain, (daily if isinstance(daily, pd.DataFrame) else pd.DataFrame())

    # 2) varre todas as abas e escolhe a melhor
    best_chain, best_score = None, -1
    # deixe um diário "corrente" válido já inicializado
    best_daily = daily if isinstance(daily, pd.DataFrame) else pd.DataFrame()
    best_daily_len = len(best_daily) if isinstance(best_daily, pd.DataFrame) else -1

    try:
        xf = pd.ExcelFile(xlsx_path_or_bytes)
        for nm in xf.sheet_names:
            try:
                ch, di = _read_chain_and_daily_matrixlike(xlsx_path_or_bytes, sheet_name=nm)
            except Exception:
                continue
            sc = _score_chain(ch)
            if sc > best_score:
                best_chain, best_score = ch, sc
            if isinstance(di, pd.DataFrame) and len(di) > best_daily_len:
                best_daily, best_daily_len = di, len(di)
    except Exception:
        pass

    if (best_chain is not None) and (best_score >= 2):
        return best_chain, (best_daily if isinstance(best_daily, pd.DataFrame) else daily)

    # 3) último fallback: ao menos devolva o diário mais “cheio”
    if isinstance(best_daily, pd.DataFrame) and not best_daily.empty:
        empty_chain = pd.DataFrame(columns=["type","strike","oi","iv","gamma","bid","ask","date"])
        return empty_chain, best_daily

    # 4) fallback final: tenta ler a 1ª aba só para OHLC
    try:
        df = pd.read_excel(xlsx_path_or_bytes, sheet_name=0, engine="openpyxl")
        empty_chain = pd.DataFrame(columns=["type","strike","oi","iv","gamma","bid","ask","date"])
        return empty_chain, _read_daily_only(df)
    except Exception as e:
        raise RuntimeError(f"Falha ao ler planilha: {e}")

def _autoscale_cents(chain: pd.DataFrame, daily: pd.DataFrame):
    """
    Detecta automaticamente planilhas em CENTAVOS e normaliza (÷100) quando necessário.
    Heurística:
      - Compara mediana dos strikes vs. último close do diário.
      - Se strike/100 ~ spot (e strike ~ 100× spot), escala STRIKE ÷100.
      - Se strike ~ spot/100 (e spot ~ 100× strike), escala DIÁRIO ÷100.
      - Fallback: se só um dos lados existir e tiver valores muito altos, divide por 100.
    Mantém a coerência de 'atr14' e 'ma200' ao escalar o diário.
    """
    ch = chain.copy() if isinstance(chain, pd.DataFrame) else pd.DataFrame()
    dy = daily.copy() if isinstance(daily, pd.DataFrame) else pd.DataFrame()

    def _median_strike(x):
        try:
            s = pd.to_numeric(x.get("strike"), errors="coerce")
            s = s[np.isfinite(s) & (s > 0)]
            return float(s.median()) if not s.empty else np.nan
        except Exception:
            return np.nan

    def _last_spot(d):
        try:
            c = pd.to_numeric(d.get("close"), errors="coerce").dropna()
            return float(c.iloc[-1]) if not c.empty else np.nan
        except Exception:
            return np.nan

    mstrike = _median_strike(ch) if not ch.empty else np.nan
    spot    = _last_spot(dy)     if not dy.empty else np.nan

    scale_chain = 1.0
    scale_daily = 1.0

    if np.isfinite(mstrike) and np.isfinite(spot) and spot > 0:
        # Quão próximo strike/100 fica do spot?
        near_chain_100 = abs((mstrike/100.0) - spot) / max(spot, 1e-9)
        near_chain_1   = abs(mstrike - spot) / max(spot, 1e-9)
        # Quão próximo strike fica de spot/100?
        near_spot_100  = abs(mstrike - (spot/100.0)) / max(spot/100.0, 1e-9) if spot >= 1e-9 else np.inf

        # Se dividir STRIKE por 100 aproxima bastante e o STRIKE atual está bem longe → STRIKE em centavos
        if (near_chain_100 < 0.25) and (near_chain_1 > 0.8):
            scale_chain = 0.01

        # Se dividir o DIÁRIO por 100 aproxima bastante e o DIÁRIO atual está bem longe → DIÁRIO em centavos
        if (near_spot_100 < 0.25) and (near_chain_1 > 0.8):
            scale_daily = 0.01

    else:
        # Fallbacks quando só um lado existe
        if not np.isfinite(spot) and np.isfinite(mstrike) and (mstrike > 2000):
            scale_chain = 0.01
        if not np.isfinite(mstrike) and np.isfinite(spot) and (spot > 2000):
            scale_daily = 0.01
        # NOVO: se AMBOS parecem ~100× e continuam coerentes depois de ÷100 → escalar os dois
    if np.isfinite(mstrike) and np.isfinite(spot) and (mstrike >= 1000) and (spot >= 1000):
        near_both_100 = abs((mstrike/100.0) - (spot/100.0)) / max(spot/100.0, 1e-9)
        if near_both_100 < 0.05:  # muito próximos após ÷100
            scale_chain = 0.01
            scale_daily = 0.01
        

    # Aplicar escalas
    if (scale_chain != 1.0) and (not ch.empty):
        ch["strike"] = pd.to_numeric(ch.get("strike"), errors="coerce") * scale_chain
        # (Opcional) se a planilha trouxer prêmio em centavos e você quiser normalizar:
        for prem_col in ["bid", "ask", "mid"]:
            if prem_col in ch.columns:
                ch[prem_col] = pd.to_numeric(ch[prem_col], errors="coerce") * scale_chain

    if (scale_daily != 1.0) and (not dy.empty):
        for col in ["open", "high", "low", "close", "ma200", "atr14"]:
            if col in dy.columns:
                dy[col] = pd.to_numeric(dy[col], errors="coerce") * scale_daily

    info = {"scale_chain": scale_chain, "scale_daily": scale_daily}
    return ch, dy, info



# ==========================
# Black–Scholes helpers
# ==========================
def _norm_pdf(x):  # φ
    return (1.0/np.sqrt(2.0*np.pi))*np.exp(-0.5*x*x)

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def _bs_price(S: float, K: float, T: float, r: float, q: float, sigma: float, opt_type: str) -> float:
    if sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if opt_type.upper().startswith("C"):
        return exp(-q * T) * S * _norm_cdf(d1) - exp(-r * T) * K * _norm_cdf(d2)
    else:
        return exp(-r * T) * K * _norm_cdf(-d2) - exp(-q * T) * S * _norm_cdf(-d1)

def _bs_gamma_grid(S_grid, K, vol, T, r=0.0, q=0.0):
    Sg = np.asarray(S_grid, dtype=float)[:, None]
    K  = np.asarray(K, dtype=float)[None, :]
    vol= np.asarray(vol, dtype=float)[None, :]
    T  = np.asarray(T, dtype=float)[None, :]
    vol = np.maximum(vol, 1e-8); T = np.maximum(T, 1e-8); Sg = np.maximum(Sg, 1e-8)
    d1 = (np.log(Sg/K) + (r - q + 0.5*vol**2)*T) / (vol*np.sqrt(T))
    return (np.exp(-q*T) * _norm_pdf(d1)) / (Sg * vol * np.sqrt(T))

# ---------- TERCEIRA SEXTA-FEIRA (OpEx BR) ----------
def _third_friday_of_month(any_date: pd.Timestamp) -> pd.Timestamp:
    d0 = pd.to_datetime(any_date).normalize().replace(day=1)
    fridays = pd.date_range(d0, d0 + pd.offsets.MonthEnd(0), freq="W-FRI")
    if len(fridays) < 3:
        d1 = (d0 + pd.DateOffset(months=1)).normalize().replace(day=1)
        fridays = pd.date_range(d1, d1 + pd.offsets.MonthEnd(0), freq="W-FRI")
    return fridays[2]

def days_to_opex(current: pd.Timestamp):
    cur = pd.to_datetime(current).normalize()
    this_tf = _third_friday_of_month(cur)
    if cur <= this_tf:
        nxt = this_tf
        prv = _third_friday_of_month(cur - pd.DateOffset(months=1))
    else:
        prv = this_tf
        nxt = _third_friday_of_month(cur + pd.DateOffset(months=1))
    dtn = (nxt - cur).days
    return dtn, dict(next=nxt, prev=prv)

def _normalize_iv_series(iv_like: pd.Series) -> pd.Series:
    # IV pode vir como "61,05%" ou "-", então parseia com to_float e detecta se está em %.
    s = pd.Series(iv_like).apply(to_float)
    if s.dropna().quantile(0.9) > 1.5:  # veio em %
        s = s / 100.0
    return s

def _infer_T_anos(df: pd.DataFrame):
    """T em anos: T_anos > DTE/252 > (referência até 3ª sexta)/252."""
    if "T_anos" in df.columns:
        return pd.to_numeric(df["T_anos"], errors="coerce")
    if "DTE" in df.columns:
        return pd.to_numeric(df["DTE"], errors="coerce").clip(lower=1)/252.0

    ref = pd.to_datetime(df.get("date", pd.NaT), errors="coerce")
    ref = ref.fillna(pd.Timestamp.today().normalize())

    def _dte_func(d):
        try:
            dd, _ = days_to_opex(d)
            if isinstance(dd, (int,float)) and np.isfinite(dd):
                return max(int(dd), 1)
        except Exception:
            pass
        return 21  # fallback ~1 mês

    dte = ref.apply(_dte_func)
    return dte/252.0

def _spot_gamma_if_needed(df: pd.DataFrame, spot: float, r=0.0, q=0.0) -> pd.Series:
    """Reprecifica Gamma no spot usando BS quando gamma está ausente/zero."""
    K   = pd.to_numeric(df.get("strike", np.nan), errors="coerce").astype(float)
    iv0 = _normalize_iv_series(df.get("iv", pd.Series(np.nan, index=df.index)))
    T   = _infer_T_anos(df).astype(float)

    iv_pos = iv0[(iv0 > 0) & np.isfinite(iv0)]
    iv_med = float(np.nanmedian(iv_pos)) if not iv_pos.empty else 0.35
    vol   = iv0.copy().astype(float).fillna(iv_med)
    vol[~(vol > 0)] = iv_med

    m = (K > 0) & (vol > 0) & (T > 0)
    G = np.zeros(len(df), dtype="float64")
    if m.any():
        Gm = _bs_gamma_grid(np.array([float(spot)]), K[m].values, vol[m].values, T[m].values, r=r, q=q)[0]
        G[m.values] = Gm
    return pd.Series(G, index=df.index, dtype="float64")

# ==========================
# Exposures (GEX) + Flip
# ==========================
def per1pct_gex(gamma, oi, spot, contract_multiplier=CONTRACT_MULTIPLIER):
    return (gamma.astype(float) * oi.astype(float) *
            float(contract_multiplier) * (float(spot)**2) * 0.01)

def compute_exposures(
    chain: pd.DataFrame,
    spot: float,
    contract_size_override: float | None = None,
    per_1pct: bool = True,
    out_unit: str = "auto",
):
    """
    GEX por strike (1% move), robusto a:
    - gamma=0/NaN (tenta BS no spot como fallback);
    - iv ausente no lado CALL (não filtra por iv>0 para o histograma);
    - cabeçalhos duplicados e vírgula decimal;
    - sinal: CALL (+), PUT (−) por convenção, independente do sinal de origem.

    Referências (paredes):
    - Call Wall = strike com **maior NET GEX** (call_gex + put_gex).
    - Put  Wall = strike com **menor NET GEX**.
    - IV Máx / IV Mín = strikes com **maior/menor IV por opção** (qualquer perna);
      se indisponível, usa média por strike como fallback.
    - Gamma Flip = raiz de **TotalGamma(S)** via **interpolação linear** entre dois S adjacentes
      onde o TotalGamma muda de sinal.
    """
    if chain is None or chain.empty:
        return pd.DataFrame(), {
            "_gex_units": "R$ / 1% move", "cov_gamma": 0.0, "cov_oi": 0.0, "cov_iv": 0.0,
            "gamma_flip": np.nan, "put_wall": np.nan, "call_wall": np.nan,
            "iv_max_strike": np.nan, "iv_min_strike": np.nan, "iv_max": np.nan, "iv_min": np.nan
        }, pd.DataFrame()

    df = chain.copy()

    # garantir colunas
    for col in ["type","strike","gamma","oi","iv","date"]:
        if col not in df.columns: df[col] = np.nan

    # normalizar
    df["type"]   = df["type"].astype(str).str.upper().str[0]
    df["strike"] = pd.to_numeric(df["strike"], errors="coerce").astype(float)
    def _conv(v):
        try:
            if isinstance(v, str):
                return to_float(v)
            return float(v)
        except Exception:
            return np.nan
    df["gamma"] = df["gamma"].apply(_conv).astype(float)
    df["oi"]    = df["oi"].apply(_conv).astype(float)
    df["iv"]    = _normalize_iv_series(df["iv"]) if "iv" in df.columns else np.nan

    # cobertura (crua)
    cov_gamma = float(np.isfinite(df["gamma"]).sum())/max(1, len(df))
    cov_oi    = float(np.isfinite(df["oi"]).sum())/max(1, len(df))
    cov_iv    = float(np.isfinite(df["iv"]).sum())/max(1, len(df))

    # parâmetros
    mult = float(contract_size_override) if contract_size_override is not None else float(CONTRACT_MULTIPLIER)
    scale_per_move = 0.01 if per_1pct else 1.0
    S = float(spot)

    # --- Fallback de gamma (BS no spot) APENAS onde gamma<=0/NaN e tivermos IV & T>0
    df["T_anos"] = _infer_T_anos(df)
    need_fallback = ~((df["gamma"] > 0) & np.isfinite(df["gamma"]))
    if need_fallback.any():
        try:
            g_bs = _spot_gamma_if_needed(df.loc[need_fallback], spot=S, r=0.0, q=0.0)
            df.loc[need_fallback, "gamma"] = g_bs.values
        except Exception:
            pass

    gamma_abs = np.abs(df["gamma"].fillna(0.0).astype(float))
    oi_val    = df["oi"].fillna(0.0).astype(float)

    # GEX por opção (no spot atual)
    df["gex_abs"] = gamma_abs * oi_val * mult * (S**2) * scale_per_move

    # agrega por strike/perna
    piv_gex = df.groupby(["strike","type"])["gex_abs"].sum().unstack().fillna(0.0)
    if piv_gex.empty:
        return pd.DataFrame(), {
            "_gex_units": "R$ / 1% move", "cov_gamma": round(100*cov_gamma,1),
            "cov_oi": round(100*cov_oi,1), "cov_iv": round(100*cov_iv,1),
            "gamma_flip": np.nan, "put_wall": np.nan, "call_wall": np.nan,
            "iv_max_strike": np.nan, "iv_min_strike": np.nan, "iv_max": np.nan, "iv_min": np.nan
        }, df

    strikes = np.array(sorted(piv_gex.index.values, key=float), dtype=float)
    call = piv_gex["C"].reindex(strikes, fill_value=0.0).values if "C" in piv_gex else np.zeros_like(strikes)
    puta = piv_gex["P"].reindex(strikes, fill_value=0.0).values if "P" in piv_gex else np.zeros_like(strikes)

    # sinal por convenção
    call_gex  = call                      # +
    put_gex   = -puta                     # −
    total_gex = call_gex + put_gex        # = call − put_abs

    # OI e IV por strike (p/ tabela/visual)
    piv_oi = df.groupby(["strike","type"])["oi"].sum().unstack().fillna(0.0)
    oi_call = piv_oi["C"].reindex(strikes, fill_value=0.0).values if "C" in piv_oi else np.zeros_like(strikes)
    oi_put  = piv_oi["P"].reindex(strikes, fill_value=0.0).values if "P" in piv_oi else np.zeros_like(strikes)

    piv_iv = df.groupby(["strike","type"])["iv"].median().unstack()
    iv_c = piv_iv["C"] if ("C" in piv_iv.columns) else pd.Series(index=strikes, dtype=float)
    iv_p = piv_iv["P"] if ("P" in piv_iv.columns) else pd.Series(index=strikes, dtype=float)
    iv_avg = pd.concat([iv_c, iv_p], axis=1).mean(axis=1).reindex(strikes).values

    # ---------- Gamma Flip: raiz de TotalGamma(S) por interpolação linear ----------
    # Avaliamos TotalGamma num grid S (uso a grade de strikes).
    # TotalGamma(S) = sum( sign(C/P) * OI * Gamma_BS(S,K,IV,T) ). (Escalas >0 não afetam a raiz.)
    try:
        K   = pd.to_numeric(df.get("strike"), errors="coerce").astype(float).values
        ivs = _normalize_iv_series(df.get("iv", pd.Series(np.nan, index=df.index))).astype(float).values
        T   = _infer_T_anos(df).astype(float).values
        typ = df.get("type").astype(str).str.upper().str[0].values
        oi0 = pd.to_numeric(df.get("oi"), errors="coerce").fillna(0.0).astype(float).values
        sgn = np.where(typ == "C", 1.0, -1.0)

        # preencher IV ausente com mediana positiva
        iv_pos = ivs[(ivs > 0) & np.isfinite(ivs)]
        iv_med = float(np.nanmedian(iv_pos)) if iv_pos.size else 0.35
        ivs_f  = np.where((ivs > 0) & np.isfinite(ivs), ivs, iv_med)

        mask = (K > 0) & (T > 0) & np.isfinite(K) & np.isfinite(T) & (oi0 > 0)
        if mask.any() and strikes.size >= 2:
            Ggrid = _bs_gamma_grid(strikes, K[mask], ivs_f[mask], T[mask])         # [len(strikes) x n_opts]
            y_tot = (Ggrid * (oi0[mask] * sgn[mask])[None, :]).sum(axis=1)         # TotalGamma(S) por S=strikes

            # Sign flip exato → raiz; senão interpola entre vizinhos com sinais opostos
            flip = np.nan
            y = np.asarray(y_tot, dtype=float)
            x = np.asarray(strikes, dtype=float)

            # se algum ponto for exatamente zero
            eq0 = np.isfinite(y) & (y == 0)
            if eq0.any():
                # pega o zero mais próximo do spot
                idx0 = int(np.nanargmin(np.abs(x[eq0] - S)))
                flip = float(x[eq0][idx0])
            else:
                s = np.sign(y)
                # índices i com s[i] * s[i+1] < 0  → mudança de sinal
                cross = np.where((s[:-1] * s[1:]) < 0)[0]
                if cross.size > 0:
                    # calcula a raiz por interpolação linear para cada cruzamento
                    cand = []
                    for i in cross:
                        x1, x2 = x[i], x[i+1]
                        y1, y2 = y[i], y[i+1]
                        if np.isfinite(y1) and np.isfinite(y2) and (y2 - y1) != 0:
                            x0 = x2 - (x2 - x1) * (y2 / (y2 - y1))
                            cand.append(float(x0))
                    if cand:
                        # escolhe o zero mais próximo do spot
                        flip = float(sorted(cand, key=lambda z: abs(z - S))[0])
                else:
                    # fallback: ponto de |TotalGamma| mínimo (aproximação)
                    i_min = int(np.nanargmin(np.abs(y)))
                    flip = float(x[i_min]) if np.isfinite(y[i_min]) else np.nan
        else:
            flip = np.nan
    except Exception:
        flip = np.nan

    # ---------- Call/Put Wall = extremos do NET GEX ----------
    if strikes.size and np.isfinite(total_gex).any():
        call_wall_val = float(strikes[int(np.nanargmax(total_gex))])  # maior NET
        put_wall_val  = float(strikes[int(np.nanargmin(total_gex))])  # menor NET
    else:
        call_wall_val = put_wall_val = np.nan

    # ---------- IV Máx / IV Mín (por OPÇÃO; call ou put) ----------
    iv_row = _normalize_iv_series(df.get("iv", pd.Series(np.nan, index=df.index)))
    K_row  = pd.to_numeric(df.get("strike"), errors="coerce").astype(float)
    mask_iv = np.isfinite(iv_row) & (iv_row > 0) & np.isfinite(K_row)

    if mask_iv.any():
        idx_max = iv_row[mask_iv].idxmax()
        idx_min = iv_row[mask_iv].idxmin()
        iv_max_strike = float(K_row.loc[idx_max])
        iv_min_strike = float(K_row.loc[idx_min])
        iv_max_val    = float(iv_row.loc[idx_max])
        iv_min_val    = float(iv_row.loc[idx_min])
    else:
        if np.isfinite(iv_avg).any():
            iv_max_strike = float(strikes[int(np.nanargmax(iv_avg))])
            iv_min_strike = float(strikes[int(np.nanargmin(iv_avg))])
            iv_max_val    = float(np.nanmax(iv_avg))
            iv_min_val    = float(np.nanmin(iv_avg))
        else:
            iv_max_strike = iv_min_strike = np.nan
            iv_max_val = iv_min_val = np.nan

    # ---------- autoescala da unidade ----------
    def _auto_scale(max_abs):
        if out_unit == "auto":
            M = float(max_abs)
            if M >= 1e9:  return 1e-9, "R$ bilhões / 1% move"
            if M >= 1e6:  return 1e-6, "R$ milhões / 1% move"
            if M >= 1e3:  return 1e-3, "R$ mil / 1% move"
            return 1.0, "R$ / 1% move"
        if str(out_unit).lower() in ["raw","none"]:
            return 1.0, "R$ / 1% move"
        return 1.0, str(out_unit)

    max_abs = np.nanmax(np.abs(np.concatenate([call_gex, put_gex, total_gex]))) if strikes.size else 0.0
    scale, unit_lbl = _auto_scale(max_abs)

    out = pd.DataFrame({
        "strike":   strikes,
        "call_gex": call_gex * scale,
        "put_gex":  put_gex  * scale,
        "total_gex":total_gex* scale,
        "oi_call":  oi_call,
        "oi_put":   oi_put,
        "iv_avg":   iv_avg,
    })

    walls = {
        "gamma_flip": float(flip) if np.isfinite(flip) else np.nan,
        "put_wall":   put_wall_val,      # menor NET
        "call_wall":  call_wall_val,     # maior NET
        "iv_max_strike": iv_max_strike,  # IV por opção (call/put)
        "iv_min_strike": iv_min_strike,
        "iv_max": iv_max_val,
        "iv_min": iv_min_val,
        "_gex_units": unit_lbl,
        "cov_gamma": round(100*cov_gamma, 1),
        "cov_oi":    round(100*cov_oi,    1),
        "cov_iv":    round(100*cov_iv,    1),
    }

    return out, walls, df

def exposures_time_series(chain: pd.DataFrame, daily: pd.DataFrame, max_days: int = 400):
    cols=["flip","net_gex_sign"]
    if ("date" not in chain.columns) or chain["date"].isna().all():
        return pd.DataFrame(index=daily.index, columns=cols)
    out=[]
    grp = chain.copy(); grp["date"]=pd.to_datetime(grp["date"], errors="coerce").dt.normalize()
    for d,g in grp.groupby("date"):
        if d not in daily.index: continue
        spot=float(daily.loc[d,"close"])
        try:
            gex_df,walls,_ = compute_exposures(g, spot)
            net=float(np.sign(gex_df["total_gex"].sum())) if not gex_df.empty else np.nan
            out.append({"date":d,"flip":walls.get("gamma_flip",np.nan),"net_gex_sign":net})
        except Exception:
            out.append({"date":d,"flip":np.nan,"net_gex_sign":np.nan})
    if not out: return pd.DataFrame(index=daily.index, columns=cols)
    df=pd.DataFrame(out).set_index("date").sort_index()
    if max_days: df=df.tail(max_days)
    return df.reindex(daily.index)

# ==========================
# Wyckoff / PRO / Fluxo / AR-GARCH / Analógicos / ML / etc.
# ==========================
def calibrate_wyckoff_params(daily: pd.DataFrame, lookback_tr: int = 20) -> dict:
    df = daily.copy()
    for c in ["open", "high", "low", "close", "vol", "atr14"]:
        if c not in df.columns: df[c] = np.nan
    dc_high = df["high"].rolling(lookback_tr, min_periods=3).max()
    dc_low  = df["low"].rolling(lookback_tr, min_periods=3).min()
    rr = (dc_high - dc_low) / (df["atr14"].replace(0, np.nan))
    rr = rr.replace([np.inf, -np.inf], np.nan).dropna()
    tr_ratio_thresh = float(np.clip(float(np.nanpercentile(rr, 40)) if not rr.empty else 1.6, 1.2, 2.4))
    upper_wick = (df["high"] - df[["open","close"]].max(axis=1)) / (df["atr14"] + 1e-9)
    lower_wick = (df[["open","close"]].min(axis=1) - df["low"])   / (df["atr14"] + 1e-9)
    w = pd.concat([upper_wick.abs(), lower_wick.abs()], axis=1).max(axis=1).replace([np.inf,-np.inf], np.nan).dropna()
    wick_z = float(np.clip(float(np.nanpercentile(w, 60)) if not w.empty else 0.45, 0.3, 0.8))
    vol = df.get("vol", pd.Series(0.0, index=df.index))
    volz = ((vol - vol.rolling(20).mean())/(vol.rolling(20).std() + 1e-9)).abs()
    ag_series = df.get("aggr", pd.Series(0.0, index=df.index))
    agz = ((ag_series - ag_series.rolling(20).mean())/(ag_series.rolling(20).std() + 1e-9)).abs()
    eff = (volz + agz).replace([np.inf, -np.inf], np.nan).dropna()
    eff_z_thresh = float(np.clip(float(np.nanpercentile(eff, 70)) if not eff.empty else 0.6, 0.4, 1.5))
    return {"tr_ratio_thresh": tr_ratio_thresh, "wick_z": wick_z, "eff_z_thresh": eff_z_thresh}

def wyckoff_signals(daily: pd.DataFrame, lookback_tr=20, tr_ratio_thresh=1.6, wick_z=0.45, eff_z_thresh=0.6):
    df=daily.copy()
    df["dc_high"]=df["high"].rolling(lookback_tr,min_periods=3).max()
    df["dc_low"] =df["low"].rolling(lookback_tr,min_periods=3).min()
    df["range_width"]=df["dc_high"]-df["dc_low"]
    df["range_ratio"]=(df["range_width"]/(df["atr14"].replace(0,np.nan))).replace([np.inf,-np.inf],np.nan)
    ma200 = df["ma200"] if "ma200" in df.columns else pd.Series(np.nan, index=df.index)
    df["trend_ma"] = np.where(df["close"] >= ma200, 1, -1)
    df.loc[ma200.isna(), "trend_ma"] = 0
    hh=(df["high"]>df["high"].shift(1))&(df["low"]>=df["low"].shift(1))
    ll=(df["low"]<df["low"].shift(1))&(df["high"]<=df["high"].shift(1))
    df["swing"]=np.where(hh,1,np.where(ll,-1,0))
    tr_flag=(df["range_ratio"]<tr_ratio_thresh)
    reg=[]
    for trg,ma,sw in zip(tr_flag, df["trend_ma"], df["swing"]):
        if trg: reg.append("TR")
        else:
            if ma>0 and sw>=0: reg.append("Markup")
            elif ma<0 and sw<=0: reg.append("Markdown")
            else: reg.append("Transição")
    df["regime"]=reg
    df["support"]=df["dc_low"]; df["resistance"]=df["dc_high"]
    volz=((df["vol"]-df["vol"].rolling(20).mean())/(df["vol"].rolling(20).std()+1e-9)).fillna(0)
    ag_series=df.get("aggr",pd.Series(0.0,index=df.index))
    agz=((ag_series-ag_series.rolling(20).mean())/(ag_series.rolling(20).std()+1e-9)).fillna(0)
    df["effort_z"]=volz.abs()+agz.abs()
    df["spring"]=False; df["ut"]=False
    for i in range(2,len(df)):
        if df["close"].iloc[i-1]<df["support"].iloc[i-1] and df["close"].iloc[i]>df["support"].iloc[i] and (ag_series.iloc[i]>0 or df["effort_z"].iloc[i]>eff_z_thresh):
            df.iloc[i, df.columns.get_loc("spring")]=True
        if df["close"].iloc[i-1]>df["resistance"].iloc[i-1] and df["close"].iloc[i]<df["resistance"].iloc[i] and (ag_series.iloc[i]<0 or df["effort_z"].iloc[i]>eff_z_thresh):
            df.iloc[i, df.columns.get_loc("ut")]=True
    upper_wick=(df["high"]-df[["open","close"]].max(axis=1))/(df["atr14"]+1e-9)
    lower_wick=(df[["open","close"]].min(axis=1)-df["low"])/(df["atr14"]+1e-9)
    df["sweep_up"]=(df["high"]>df["resistance"])&(df["close"]<df["resistance"])&(upper_wick>wick_z)&(agz<0)
    df["sweep_dn"]=(df["low"]<df["support"])&(df["close"]>df["support"])&(lower_wick>wick_z)&(agz>0)
    mid=(df["support"]+df["resistance"])/2; phase=[]
    for i in range(len(df)):
        if df["regime"].iloc[i]=="TR": phase.append("ACC" if df["close"].iloc[i] <= mid.iloc[i] else "DST")
        elif df["regime"].iloc[i]=="Markup": phase.append("MRKP")
        elif df["regime"].iloc[i]=="Markdown": phase.append("MRKD")
        else: phase.append("TRANS")
    df["phase"]=phase
    score=[]
    for i in range(len(df)):
        s=50
        s+=10 if df["regime"].iloc[i]=="Markup" else 0
        s-=10 if df["regime"].iloc[i]=="Markdown" else 0
        s+=np.sign((df["close"].iloc[i]-df["close"].shift(1).iloc[i]))*5
        if df["spring"].iloc[i] or df["sweep_dn"].iloc[i]: s+=10
        if df["ut"].iloc[i] or df["sweep_up"].iloc[i]: s-=10
        score.append(np.clip(s,0,100))
    df["wyckoff_score"]=score
    return df

def smooth_wyckoff_blocks(wy_df: pd.DataFrame, min_len=3):
    ph=wy_df["phase"].astype(str); runs=_runs_of_equals(ph); to_fix=[r for r in runs if r[3]<min_len]
    if not to_fix: return wy_df
    ph=ph.copy()
    for x0,x1,val,ln in to_fix:
        idx0=ph.index.get_loc(x0); idx1=ph.index.get_loc(x1)
        i=runs.index((x0,x1,val,ln))
        left=runs[i-1] if i>0 else None; right=runs[i+1] if i<len(runs)-1 else None
        rep = left[2] if (left and (not right or left[3]>=right[3])) else (right[2] if right else None)
        if rep: ph.iloc[idx0:idx1+1]=rep
    out=wy_df.copy(); out["phase"]=ph.values; return out

def phase_label_pt(wy_df: pd.DataFrame) -> pd.Series:
    lab=[]
    for i in range(len(wy_df)):
        ph=str(wy_df["phase"].iloc[i]); prev_reg="TR"
        if i>0:
            prev_slice=wy_df["regime"].iloc[max(0,i-10):i]
            if not prev_slice.empty: prev_reg=prev_slice.mode().iloc[0]
        if ph=="ACC":  lab.append("Acumulação" if prev_reg!="Markup" else "Reacumulação")
        elif ph=="DST":lab.append("Distribuição" if prev_reg!="Markdown" else "Redistribuição")
        elif ph=="MRKP": lab.append("Tendência de Alta")
        elif ph=="MRKD": lab.append("Tendência de Baixa")
        elif ph=="TRANS": lab.append("Transição")
        else: lab.append(ph)
    out=wy_df.copy(); out["fase_pt"]=lab; return out["fase_pt"]

def wyckoff_pro_enhance(daily: pd.DataFrame, wy_df: pd.DataFrame) -> pd.DataFrame:
    df = daily.copy()
    out = wy_df.copy()
    rng   = (df["high"] - df["low"]).astype(float)
    body  = (df["close"] - df["open"]).abs().astype(float)
    close_pos = (df["close"] - df["low"]) / (rng + 1e-9)
    ret1  = np.log(df["close"]).diff()
    vol    = df.get("vol", pd.Series(0.0, index=df.index)).astype(float).fillna(0.0)
    vol_z  = ((vol - vol.rolling(20).mean())/(vol.rolling(20).std() + 1e-9)).fillna(0.0)
    ag     = df.get("aggr", pd.Series(0.0, index=df.index)).astype(float).fillna(0.0)
    ag_z   = ((ag - ag.rolling(20).mean())/(ag.rolling(20).std() + 1e-9)).fillna(0.0)
    atr    = df["high"] - df["low"]
    atr_ma = atr.rolling(20).mean()
    z_hi = vol_z.quantile(0.80)
    z_lo = vol_z.quantile(0.30)
    wide = (rng > 1.6*atr_ma)
    narrow = (rng < 0.8*atr_ma)
    climax_up   = (vol_z > z_hi) & wide & (close_pos > 0.7) & (ret1 > 0)
    climax_down = (vol_z > z_hi) & wide & (close_pos < 0.3) & (ret1 < 0)
    churn = (vol_z > z_hi) & (body < 0.35*rng)
    up_trend   = ret1.rolling(5).sum() > 0
    down_trend = ret1.rolling(5).sum() < 0
    no_demand = up_trend & (ret1 > 0) & narrow & (vol_z < z_lo) & (close_pos > 0.5)
    no_supply = down_trend & (ret1 < 0) & narrow & (vol_z < z_lo) & (close_pos < 0.5)
    sup = out["support"].astype(float)
    res = out["resistance"].astype(float)
    sos = (df["close"] > res) & (vol_z > z_hi*0.9)
    sow = (df["close"] < sup) & (vol_z > z_hi*0.9)
    lps = pd.Series(False, index=df.index)
    for i in range(2, len(df)):
        if sos.iloc[i-2] or (i >= 3 and sos.iloc[i-3]):
            atr_i = float(atr_ma.iloc[i])
            if not np.isfinite(atr_i) or atr_i <= 0:
                atr_i = 1e-9
            near_old_res = abs(df["low"].iloc[i] - res.iloc[i-2]) <= (0.5 * atr_i)
            lps.iloc[i] = near_old_res and (vol_z.iloc[i] < z_lo) and (df["close"].iloc[i] > df["open"].iloc[i])
    utad = pd.Series(False, index=df.index)
    for i in range(3, len(df)):
        broke = (df["high"].iloc[i-2] > res.iloc[i-2]) and (df["close"].iloc[i-2] < res.iloc[i-2])
        gave_back = (df["close"].iloc[i] < res.iloc[i]) and (vol_z.iloc[i-1] > z_hi*0.8)
        utad.iloc[i] = broke and gave_back
    choch = (rng > 1.8*atr_ma) & (np.sign(ret1).rolling(3).sum().abs() <= 1)
    base = out.get("wyckoff_score", pd.Series(50, index=df.index)).astype(float)
    score = base +  \
            8.0*(sos | lps | no_supply | out.get("spring", False))  - \
            8.0*(sow | utad | no_demand | out.get("ut", False))     + \
            4.0*climax_up - 4.0*climax_down + \
            2.0*churn + 3.0*choch.astype(float)
    out["climax_up"] = climax_up.values
    out["climax_down"] = climax_down.values
    out["churn"] = churn.values
    out["no_demand"] = no_demand.values
    out["no_supply"] = no_supply.values
    out["sos"] = sos.values
    out["sow"] = sow.values
    out["lps"] = lps.values
    out["utad"] = utad.values
    out["choch"] = choch.values
    out["wy_score_pro"] = score.clip(0, 100).values
    return out

def _ema(s, span): return pd.Series(s, index=s.index).ewm(span=span, adjust=False).mean()
def _rsi(close, period=14):
    diff=close.diff(); up=diff.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    dn=(-diff.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs=up/(dn+1e-9); return 100-(100/(1+rs))
def _macd_hist(close):
    macd=_ema(close,12)-_ema(close,26); sig=macd.ewm(span=9, adjust=False).mean(); return macd-sig
def _bb_pos(close, length=20):
    ma=close.rolling(length).mean(); sd=close.rolling(length).std(); return (close-ma)/(sd+1e-9)

def spot_player_features(daily: pd.DataFrame, wy_df: pd.DataFrame, lookback_corr=30):
    df=daily.copy(); df["ret1d"]=np.log(df["close"]).diff(); df["tr"]=(df["high"]-df["low"]).clip(lower=1e-9)
    vol=df["vol"].fillna(0.0); ag=df.get("aggr",pd.Series(0.0,index=df.index))
    has_aggr = pd.api.types.is_numeric_dtype(ag) and np.isfinite(ag).sum()>0 and ag.abs().sum()>1e-6
    if has_aggr:
        ag=ag.fillna(0.0)
        df["ag_z"]=(ag-ag.rolling(60,min_periods=20).mean())/(ag.rolling(60,min_periods=20).std()+1e-9)
        cvd=ag.cumsum()
    else:
        ret=df["close"].pct_change().fillna(0.0)
        cvd=(np.sign(ret)*vol).cumsum()
        df["ag_z"]=(vol-vol.rolling(60,min_periods=20).mean())/(vol.rolling(60,min_periods=20).std()+1e-9)
    cvd_delta=cvd.diff(5).rolling(5,min_periods=3).mean(); cvd_std=cvd.rolling(60,min_periods=20).std()+1e-9
    df["cvd_chg_z"]=(cvd_delta/cvd_std).fillna(0.0)
    pr_10=df["ret1d"].rolling(10).sum(); cvd_10=cvd.diff(10)
    df["bull_div"]=(pr_10<0)&(cvd_10>0); df["bear_div"]=(pr_10>0)&(cvd_10<0)
    df["absorb_up"]=(df["ag_z"]>1.0)&(df["ret1d"].rolling(3).sum()<=0)&(df["tr"]<df["atr14"])
    df["absorb_dn"]=(df["ag_z"]<-1.0)&(df["ret1d"].rolling(3).sum()>=0)&(df["tr"]<df["atr14"])
    raw=0.9*df["cvd_chg_z"].iloc[-1] + 0.4*df["ag_z"].iloc[-1]; player_dir=float(np.tanh(raw))
    diag={"player_dir":player_dir,"ag_z":float(df["ag_z"].iloc[-1]),"cvd_slope":float(df["cvd_chg_z"].iloc[-1]),
          "flags":{"bull_divergence":bool(df["bull_div"].iloc[-1]),"bear_divergence":bool(df["bear_div"].iloc[-1]),
                   "absorption_up":bool(df["absorb_up"].iloc[-1]),"absorption_dn":bool(df["absorb_dn"].iloc[-1])},
          "source":"aggr" if has_aggr else "obv"}
    return df, diag

def garch_arma_signals(returns: pd.Series, chain: pd.DataFrame, horizon: int = 10):
    try:
        r=pd.Series(returns,dtype="float64").dropna()
    except Exception:
        r=pd.Series([],dtype="float64")
    r=r.tail(1500).reset_index(drop=True)
    if len(r)<80:
        return {"signal":"no-trade","reason":"few data","sigma_1d":np.nan,"sigma_ann":np.nan,"mu_1d":0.0,
                "IV_front":np.nan,"IVRV":np.nan,"bias":"flat","strategy":"n/a","sig_path":[], "sig_cum_path":[]}
    r_scaled=r*100.0
    try:
        arma=ARIMA(r_scaled, order=(1,0,0)).fit()
        mu1d=float(np.asarray(arma.get_forecast(steps=1).predicted_mean).squeeze())/100.0
    except Exception:
        mu1d=float(r.mean())
    try:
        res=arch_model(r_scaled, mean="Zero", vol="GARCH", p=1, q=1, dist="t").fit(disp="off")
        fvar=res.forecast(horizon=horizon, reindex=False).variance.values[-1]
        fvar=np.asarray(fvar,dtype="float64"); fvar=np.maximum(fvar,0.0)
        sig_path=np.sqrt(fvar)/100.0; sig1d=float(sig_path[0]); sig_cum=np.sqrt(np.cumsum(fvar))/100.0
    except Exception:
        sig_hist=float(r.std()); sig_hist= sig_hist if (np.isfinite(sig_hist) and sig_hist>0) else 0.02
        sig_path=np.full(int(horizon), sig_hist, dtype="float64"); sig1d=float(sig_path[0])
        sig_cum=sig_hist*np.sqrt(np.arange(1, int(horizon)+1, dtype="float64"))
    sig_ann=float(sig1d*np.sqrt(252.0)) if np.isfinite(sig1d) else np.nan
    try:
        iv_front = safe_median(pd.Series(chain.get("iv", np.nan), dtype="float64").replace(0.0,np.nan)) \
            if isinstance(chain, pd.DataFrame) and "iv" in chain.columns else np.nan
    except Exception: iv_front=np.nan
    ivrv = float(iv_front)/float(sig_ann) if np.isfinite(iv_front) and np.isfinite(sig_ann) and sig_ann>0 else np.nan
    bias="up" if mu1d>0 else ("down" if mu1d<0 else "flat")
    return {"signal":"ok","sigma_1d":float(sig1d),"sigma_ann":float(sig_ann),"mu_1d":float(mu1d),
            "IV_front":float(iv_front) if np.isfinite(iv_front) else np.nan,
            "IVRV":float(ivrv) if np.isfinite(ivrv) else np.nan, "bias":bias,"strategy":"n/a",
            "sig_path":[float(x) for x in np.asarray(sig_path)], "sig_cum_path":[float(x) for x in np.asarray(sig_cum)]}

def forecast_band_ar_garch(daily: pd.DataFrame, ga: dict, horizon: int = 10, last_n: int = 200):
    close = daily["close"].dropna()
    if len(close) < 20:
        raise ValueError("Série muito curta.")

    hist = close.iloc[-last_n:]
    S0   = float(hist.iloc[-1])

    mu = float(ga.get("mu_1d", 0.0))
    sig_cum = np.array(ga.get("sig_cum_path", []), dtype=float)

    if sig_cum.size < horizon:
        sig1d = float(ga.get("sigma_1d", np.nan))
        if not (np.isfinite(sig1d) and sig1d > 0):
            # fallback em caso de GA sem saída
            sig1d = float(close.pct_change().dropna().tail(252).std() or 0.02)
        sig_cum = sig1d * np.sqrt(np.arange(1, horizon + 1, dtype=float))

    last = close.index[-1]
    try:
        fwd_dates = pd.bdate_range(start=last, periods=horizon + 1, inclusive="right")
    except TypeError:
        try:
            fwd_dates = pd.bdate_range(start=last, periods=horizon + 1, closed="right")
        except TypeError:
            fwd_dates = pd.bdate_range(start=last + BDay(1), periods=horizon)

    n = min(horizon, len(fwd_dates))
    t = np.arange(1, n + 1, dtype=float)

    mean_path = S0 * np.exp(mu * t)
    band      = sig_cum[:n]

    up1 = S0 * np.exp(mu * t + band)
    dn1 = S0 * np.exp(mu * t - band)

    proj = pd.DataFrame({"date": fwd_dates[:n], "mean": mean_path, "up1": up1, "dn1": dn1}).set_index("date")
    return hist, proj

def expected_move_days(ga: dict, spot: float, days: int = 10):
    sig_cum=np.array(ga.get("sig_cum_path",[]),dtype=float)
    if sig_cum.size<days:
        s1d=float(ga.get("sigma_1d",np.nan))
        if not (np.isfinite(s1d) and s1d>0): return np.nan,np.nan
        sig_cum = s1d*np.sqrt(np.arange(1,days+1,dtype=float))
    sc=float(sig_cum[min(days-1,len(sig_cum)-1)])
    em_pct=(np.exp(sc)-1.0)*100.0; em_abs=spot*(np.exp(sc)-1.0)
    return em_abs, em_pct

def _auto_arima_fallback(y, seasonal=True, m=21, max_p=2, max_q=2, max_P=1, max_Q=1):
    y = pd.Series(y).dropna()
    best = None
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            if seasonal:
                for P in range(max_P + 1):
                    for Q in range(max_Q + 1):
                        order = (p, 0, q)
                        sorder = (P, 0, Q, m)
                        try:
                            res = SARIMAX(
                                y, order=order, seasonal_order=sorder,
                                enforce_stationarity=False, enforce_invertibility=False
                            ).fit(disp=False)
                            aic = res.aic
                            if (best is None) or (aic < best[0]):
                                best = (aic, res)
                        except Exception:
                            pass
            else:
                order = (p, 0, q)
                try:
                    res = SARIMAX(
                        y, order=order, seasonal_order=(0, 0, 0, 0),
                        enforce_stationarity=False, enforce_invertibility=False
                    ).fit(disp=False)
                    aic = res.aic
                    if (best is None) or (aic < best[0]):
                        best = (aic, res)
                except Exception:
                    pass

    if best is None:
        raise RuntimeError("auto-arima fallback não convergiu")

    class _Model:
        def __init__(self, res): self.res = res
        def predict(self, n_periods=1):
            fc = self.res.get_forecast(steps=n_periods).predicted_mean
            return np.asarray(fc)
    return _Model(best[1])

PHASE_TO_CODE={"ACC":0,"DST":1,"MRKP":2,"MRKD":3,"TRANS":4,"TR":5}
def _robust_scale(col: pd.Series):
    a=np.asarray(col.values,dtype="float64"); mask=np.isfinite(a)
    if mask.sum()==0: return np.zeros_like(a),0.0,1.0
    a_val=a[mask]; med=float(np.nanmedian(a_val)); mad=float(np.nanmedian(np.abs(a_val-med)))
    if not np.isfinite(mad) or mad<1e-12:
        std=float(np.nanstd(a_val)); mad=std if (np.isfinite(std) and std>=1e-12) else 1.0
    z=np.zeros_like(a,dtype="float64"); z[mask]=(a[mask]-med)/mad; z[~np.isfinite(z)]=0.0
    return z,med,mad

def _weighted_quantile(values, quantiles, weights=None):
    values=np.asarray(values,dtype="float64"); q=np.asarray(quantiles,dtype="float64")
    if values.size==0: return np.full_like(q,np.nan,dtype="float64")
    if weights is None: return np.quantile(values,q)
    weights=np.asarray(weights,dtype="float64")
    if weights.sum()<=0 or not np.isfinite(weights).any(): return np.quantile(values,q)
    sorter=np.argsort(values); v=values[sorter]; w=weights[sorter]
    cw=np.cumsum(w); cw=cw/ (cw[-1] if cw[-1] else 1.0)
    return np.interp(q,cw,v)

def build_state_features(daily: pd.DataFrame, wy: pd.DataFrame, flow_df: pd.DataFrame,
                         expo_ts: pd.DataFrame | None = None) -> pd.DataFrame:
    S=pd.DataFrame(index=daily.index); close=daily["close"]
    S["ret1"]=np.log(close).diff(); S["ret5"]=np.log(close).diff(5)
    S["atr_pct"]=(daily["atr14"]/close).replace([np.inf,-np.inf],np.nan)
    S["rng_atr"]=((daily["high"]-daily["low"])/(daily["atr14"]+1e-9)).clip(0,10)
    S["phase_code"]=wy["phase"].map(PHASE_TO_CODE).astype(float)
    S["is_markup"]=(wy["regime"]=="Markup").astype(float)
    S["is_markdown"]=(wy["regime"]=="Markdown").astype(float)
    S["spring"]=wy["spring"].astype(float); S["ut"]=wy["ut"].astype(float)
    S["sweep_up"]=wy["sweep_up"].astype(float); S["sweep_dn"]=wy["sweep_dn"].astype(float)
    S["dist_sup_atr"]=(close-wy["support"])/(daily["atr14"]+1e-9)
    S["dist_res_atr"]=(wy["resistance"]-close)/(daily["atr14"]+1e-9)
    for c in ["ag_z","cvd_chg_z","absorb_up","absorb_dn","bull_div","bear_div"]:
        if c in flow_df.columns: S[c]=flow_df[c].astype(float)
    S["rsi14"]=_rsi(daily["close"]).clip(0,100); S["macd_hist"]=_macd_hist(daily["close"]); S["bb_pos"]=_bb_pos(daily["close"])
    if expo_ts is not None and not expo_ts.empty:
        S["dist_flip_atr"]=(close-expo_ts["flip"])/(daily["atr14"]+1e-9)
        S["dealer_sign"]=expo_ts["net_gex_sign"]
    return S

def find_analogs(S: pd.DataFrame, close: pd.Series, horizon: int = 10, k: int = 50, feature_weights: dict | None = None):
    idx=S.index; fwd=np.log(close.shift(-horizon)/close)
    Z=pd.DataFrame(index=idx)
    for c in S.columns: Z[c]=_robust_scale(S[c])[0]
    if feature_weights is None:
        feature_weights={"ret1":1.0,"ret5":0.8,"atr_pct":0.7,"rng_atr":0.6,"phase_code":0.8,"is_markup":0.7,"is_markdown":0.7,
                         "spring":0.8,"ut":0.8,"sweep_up":0.5,"sweep_dn":0.5,"dist_sup_atr":0.9,"dist_res_atr":0.9,
                         "ag_z":1.0,"cvd_chg_z":1.0,"absorb_up":0.6,"absorb_dn":0.6,"bull_div":0.6,"bear_div":0.6,
                         "rsi14":0.6,"macd_hist":0.7,"bb_pos":0.6,"dist_flip_atr":0.6,"dealer_sign":0.5}
    used_cols=[c for c in Z.columns if c in feature_weights]; w=np.array([feature_weights[c] for c in used_cols],dtype=float)
    info_mask=np.array([not np.allclose(Z[c].values,0.0) for c in used_cols]); used_cols=[c for c,k in zip(used_cols,info_mask) if k]; w=w[info_mask]
    if len(used_cols)==0:
        return {"prob_up":np.nan,"ret_mean":np.nan,"ret_q20":np.nan,"ret_q80":np.nan,"em_pct_ana":np.nan,"analog_dir":0.0,"matches":pd.DataFrame()}
    x0z=Z.iloc[-1][used_cols].values; mask_valid=np.isfinite(x0z); valid_rows=np.where(fwd.notna().values[:-1])[0]
    D=[]
    for i in valid_rows:
        xi=Z.iloc[i][used_cols].values; m=np.isfinite(xi)&mask_valid
        if not m.any(): continue
        d=np.sqrt(np.sum(((x0z[m]-xi[m])*w[m])**2)/np.sum(w[m]**2)); D.append((i,d))
    if not D:
        return {"prob_up":np.nan,"ret_mean":np.nan,"ret_q20":np.nan,"ret_q80":np.nan,"em_pct_ana":np.nan,"analog_dir":0.0,"matches":pd.DataFrame()}
    D.sort(key=lambda t: t[1]); sel=D[:min(k,len(D))]; idxs=[i for i,_ in sel]; dists=np.array([d for _,d in sel])
    sim=1.0/(1.0+dists); sim=sim/sim.sum(); r=fwd.iloc[idxs].values
    prob_up=float(np.sum(sim*(r>0))); ret_mean=float(np.sum(sim*r)); ret_q20,ret_q80=_weighted_quantile(r,[0.2,0.8],weights=sim)
    em_pct_ana=float((np.exp(max(abs(ret_q20),abs(ret_q80)))-1.0)*100.0); analog_dir=float(np.clip(2*prob_up-1.0,-1.0,1.0))
    matches=pd.DataFrame({"date":S.index.values[idxs],"dist":dists,"weight":sim,"fwd":r}).sort_values("dist")
    return {"prob_up":prob_up,"ret_mean":ret_mean,"ret_q20":float(ret_q20),"ret_q80":float(ret_q80),
            "em_pct_ana":em_pct_ana,"analog_dir":analog_dir,"matches":matches}

def ml_directional_prob(S: pd.DataFrame, close: pd.Series, horizon: int = 10, min_train: int = 250):
    y=(np.log(close.shift(-horizon)/close)>0).astype(int).rename("y")
    X0=S.copy().replace([np.inf,-np.inf],np.nan).ffill(limit=10).bfill(limit=10).fillna(0.0)
    df=pd.concat([X0,y],axis=1).dropna()
    if len(df)<120: return {"prob_up_ml":np.nan,"auc_cv":np.nan,"coef":{}}
    dyn_min=min(250,max(60,int(0.30*len(df)))); min_train=dyn_min
    X,yv=df.drop(columns=["y"]),df["y"]; tscv=TimeSeriesSplit(n_splits=5); oof=np.full(len(yv),np.nan)
    for tr,te in tscv.split(X):
        if len(tr)<min_train: continue
        pipe=Pipeline([("sc",StandardScaler(with_mean=False)),("lr",LogisticRegression(max_iter=400,class_weight="balanced",solver="lbfgs"))])
        pipe.fit(X.iloc[tr].values, yv.iloc[tr].values); oof[te]=pipe.predict_proba(X.iloc[te].values)[:,1]
    auc = np.nan
    mask = ~np.isnan(oof)
    if mask.any():
        y_true = yv[mask].values
        p_hat  = oof[mask]
        try:
            if np.unique(y_true).size >= 2:
                auc = float(roc_auc_score(y_true, p_hat))
        except Exception:
            auc = np.nan
    cutoff=-horizon if horizon>0 else len(X)
    pipe=Pipeline([("sc",StandardScaler(with_mean=False)),("lr",LogisticRegression(max_iter=400,class_weight="balanced",solver="lbfgs"))])
    pipe.fit(X.iloc[:cutoff].values, yv.iloc[:cutoff].values)
    prob=float(pipe.predict_proba(X.iloc[[-1]].values)[:,1][0]); coef=dict(zip(X.columns, pipe.named_steps["lr"].coef_.ravel()))
    return {"prob_up_ml":prob,"auc_cv":auc,"coef":coef}

def drift_family(close: pd.Series, horizon: int = 1, seasonal_m: int = 21):
    ret=pd.Series(np.log(close).diff(),dtype="float64").dropna().reset_index(drop=True)
    out={"mu_ar1":np.nan,"mu_sarima":np.nan,"mu_auto":np.nan}
    try:
        arma=ARIMA(ret*100, order=(1,0,0)).fit(); out["mu_ar1"]=float(arma.get_forecast(steps=1).predicted_mean.iloc[0]/100.0)
    except Exception:
        out["mu_ar1"]=float(ret.mean())
    try:
        if _auto_arima is not None:
            mdl=_auto_arima(ret, seasonal=True, m=seasonal_m, error_action="ignore", suppress_warnings=True)
            out["mu_auto"]=float(mdl.predict(n_periods=1)[0])
        else:
            mdl=_auto_arima_fallback(ret, seasonal=True, m=seasonal_m)
            out["mu_auto"]=float(mdl.predict(n_periods=1)[0])
    except Exception:
        pass
    try:
        sar=SARIMAX(ret, order=(1,0,1), seasonal_order=(1,0,1,seasonal_m),
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        out["mu_sarima"]=float(sar.get_forecast(steps=1).predicted_mean.iloc[0])
    except Exception:
        pass
    return out

def linear_return_mu(S: pd.DataFrame, close: pd.Series, horizon: int = 10):
    y=np.log(close.shift(-horizon)/close); X=S.replace([np.inf,-np.inf],np.nan).ffill(limit=10).bfill(limit=10).fillna(0.0)
    df=pd.concat([X,y.rename("y")],axis=1).dropna()
    if len(df)<120: return {"mu_lin":np.nan}
    cutoff=-horizon if horizon>0 else len(df)
    mdl=LinearRegression(); mdl.fit(df.iloc[:cutoff].drop(columns=["y"]).values, df.iloc[:cutoff]["y"].values)
    mu=float(mdl.predict(df.iloc[[-1]].drop(columns=["y"]).values)[0]); return {"mu_lin":mu}

def lstm_directional_prob(close: pd.Series, horizon: int = 10, lookback: int = 30):
    if not _HAS_TF: return {"prob_up_dl":np.nan}
    r=np.log(close).diff().dropna().values.astype("float32")
    if r.shape[0]<(lookback+120): return {"prob_up_dl":np.nan}
    X,y=[],[]
    for i in range(lookback, len(r)-horizon):
        X.append(r[i-lookback:i]); y.append(1.0 if r[i:i+horizon].sum()>0 else 0.0)
    import tensorflow as tf  # noqa
    from tensorflow import keras  # noqa
    X=np.array(X)[...,None]; y=np.array(y,dtype="float32"); split=int(0.8*len(X))
    Xtr,ytr,Xte,yte=X[:split],y[:split],X[split:],y[split:]
    mdl=keras.Sequential([keras.layers.Input(shape=(lookback,1)), keras.layers.LSTM(16), keras.layers.Dense(1,activation="sigmoid")])
    mdl.compile(optimizer="adam", loss="binary_crossentropy"); mdl.fit(Xtr,ytr,epochs=5,batch_size=32,verbose=0,validation_data=(Xte,yte))
    p=float(mdl.predict(X[[-1]], verbose=0)[0][0]); return {"prob_up_dl":p}

def kelly_dir(mu_1d: float, sigma_1d: float):
    if not (np.isfinite(mu_1d) and np.isfinite(sigma_1d) and sigma_1d>0): return {"rl_dir":0.0,"kelly":np.nan}
    f=mu_1d/(sigma_1d**2 + 1e-12); f=float(np.clip(f,-1.0,1.0)); return {"rl_dir":f,"kelly":f}

def _dir_from_prob(p): return float(np.clip(2.0*p-1.0,-1.0,1.0)) if np.isfinite(p) else 0.0
def _dir_from_mu(mu, sig):
    if not np.isfinite(mu): return 0.0
    z = mu/(sig+1e-9) if np.isfinite(sig) and sig>0 else mu*10.0
    return float(np.clip(z,-1.0,1.0))

def joint_directional_view_hybrid(wy_last: pd.Series, walls: dict, spot: float,
                                  df_gex: pd.DataFrame, ga: dict, analog: dict, ml: dict,
                                  drifts=None, lin=None, dl=None, rl=None):
    wy_dir=0.0
    if wy_last.get("regime")=="Markup": wy_dir+=0.7
    if wy_last.get("regime")=="Markdown": wy_dir-=0.7
    if bool(wy_last.get("spring",False)) or bool(wy_last.get("sweep_dn",False)): wy_dir+=0.3
    if bool(wy_last.get("ut",False)) or bool(wy_last.get("sweep_up",False)): wy_dir-=0.3
    wy_dir=float(np.clip(wy_dir,-1,1))
    if df_gex is not None and not df_gex.empty:
        net_gex = float(df_gex["total_gex"].sum()) if "total_gex" in df_gex.columns else 0.0
    else:
        net_gex = 0.0
    flip=(walls or {}).get("gamma_flip",np.nan); atr=float(wy_last.get("atr14",np.nan))
    prox=np.exp(-abs(spot-float(flip))/(1.5*atr)) if (np.isfinite(atr) and np.isfinite(flip)) else 1.0
    gex_dir=0.0
    if np.isfinite(flip): gex_dir=(np.sign(spot-float(flip)) if net_gex<0 else -np.sign(spot-float(flip)))*prox
    gex_dir=float(np.clip(gex_dir,-1,1))
    analog_dir=float(analog.get("analog_dir",0.0))
    ml_dir=_dir_from_prob(float(ml.get("prob_up_ml",np.nan)))
    sig=float(ga.get("sigma_1d",np.nan)); mu_lin=float((lin or {}).get("mu_lin",np.nan))
    lin_dir=_dir_from_mu(mu_lin, sig)
    d=drifts or {}; mu_sar=d.get("mu_auto",np.nan); mu_sar = mu_sar if np.isfinite(mu_sar) else d.get("mu_sarima",np.nan)
    sar_dir=_dir_from_mu(float(mu_sar), sig)
    prob_dl=float((dl or {}).get("prob_up_dl",np.nan)); dl_dir=_dir_from_prob(prob_dl)
    rl_dir=float((rl or {}).get("rl_dir",0.0))
    w=dict(wy=0.22,gex=0.15,ana=0.20,ml=0.18,lin=0.10,sar=0.10,dl=0.03,rl=0.02)
    score=np.clip(w["wy"]*wy_dir + w["gex"]*gex_dir + w["ana"]*analog_dir + w["ml"]*ml_dir + w["lin"]*lin_dir + w["sar"]*sar_dir + w["dl"]*dl_dir + w["rl"]*rl_dir, -1,1)
    signal="UP" if score>0.15 else ("DOWN" if score<-0.15 else "NEUTRAL")
    conf=int(np.clip(40+60*abs(score),0,100))
    return {"signal":signal,"score":round(float(score),3),"confidence":conf,
            "parts":{"wyckoff_dir":round(wy_dir,3),"gex_dir":round(gex_dir,3),"analog_dir":round(analog_dir,3),
                     "ml_dir":round(ml_dir,3),"lin_dir":round(lin_dir,3),"sarima_dir":round(sar_dir,3),
                     "dl_dir":round(dl_dir,3),"rl_dir":round(rl_dir,3),
                     "dealer_regime":"PosGamma" if net_gex>=0 else "NegGamma","flip":float(flip) if np.isfinite(flip) else None}}

def iv_front_series(chain: pd.DataFrame):
    if "date" not in chain.columns or "iv" not in chain.columns: return pd.Series(dtype=float)
    t = chain.copy(); t["iv"] = _normalize_iv_series(t["iv"])
    iv_ts = t.groupby(pd.to_datetime(t["date"]).dt.normalize())["iv"].median().sort_index()
    return iv_ts

def iv_percentile(iv_ts: pd.Series, lookback=252):
    if iv_ts.empty: return np.nan
    iv_ts = iv_ts.dropna()
    if iv_ts.empty: return np.nan
    window = iv_ts.tail(lookback)
    return float((window.rank(pct=True).iloc[-1])*100.0)

def _num(x, nd="n/d", fmt="{:.2f}"):
    try:
        xx=float(x)
        if not np.isfinite(xx): return nd
        return fmt.format(xx)
    except Exception: return nd

def _prob_blend(joint_score: float, p_ml: float, p_ana: float, p_dl: float = np.nan) -> float:
    comps = []
    if np.isfinite(p_ml):  comps.append(("ml",  p_ml, 0.45))
    if np.isfinite(p_ana): comps.append(("ana", p_ana, 0.30))
    if np.isfinite(joint_score):
        p_ens = 0.5 + 0.5*float(np.clip(joint_score, -1, 1))
        comps.append(("ens", p_ens, 0.20))
    if np.isfinite(p_dl):  comps.append(("dl",  p_dl, 0.05))
    w = sum(x[2] for x in comps)
    return float(sum(p*w for _,p,w in comps)/w) if w>0 else np.nan

def _blended_move_band(daily: pd.DataFrame, ga: dict, analog: dict, days: int = 10):
    S0 = float(daily["close"].iloc[-1])
    em_abs_g, em_pct_g = expected_move_days(ga, S0, days=10)
    r20, r80 = analog.get("ret_q20", np.nan), analog.get("ret_q80", np.nan)
    em_pct_a = (np.exp(max(abs(r20), abs(r80))) - 1.0)*100.0 if np.isfinite(r20) and np.isfinite(r80) else np.nan
    em_pct = float(np.nanmax([em_pct_g, em_pct_a]))
    if not np.isfinite(em_pct): 
        return np.nan, np.nan, np.nan
    delta = S0 * em_pct/100.0
    return S0 - delta, S0 + delta, em_pct

DEFAULT_PARAMS = {
    "A": {"flip_atr":0.5, "ivrv":1.15, "theta_frac":0.55, "rv_stop":1.2},
    "B": {"ag_z":1.0, "rng_lookback":20, "R_mult_tp":1.2, "trail_len":3},
    "C": {"days_pre":[1,3]},
    "D": {"days_post":[1,3]},
}

def playbook_signal(r: dict, params: dict=DEFAULT_PARAMS):
    daily=r["daily"]; wy=r["wy"]; wy_last=r["wy_last"]; ga=r["ga"]; walls=r["walls"]; spot=r["spot"]
    flow=r["flow"]; expo_ts = r["expo_ts"]
    iv_ts = iv_front_series(r["chain"]) if "chain" in r else iv_front_series(pd.DataFrame())
    iv_pct = iv_percentile(iv_ts) if iv_ts is not None and not iv_ts.empty else np.nan
    ivrv = ga.get("IVRV", np.nan)
    flip=walls.get("gamma_flip",np.nan)
    dist_flip_atr = abs(spot - float(flip))/float(daily["atr14"].iloc[-1]) if (np.isfinite(flip) and np.isfinite(daily["atr14"].iloc[-1]) and daily["atr14"].iloc[-1]>0) else np.nan
    dealer=r.get("dealer","PosGamma")
    ag_z=float(flow.get("ag_z", np.nan))
    is_tr = wy_last.get("regime") in ["TR","Transição"]
    rec=None; rationale=[]
    if (dealer=="PosGamma") and is_tr and (np.isfinite(dist_flip_atr) and dist_flip_atr>params["A"]["flip_atr"]) and (np.isfinite(ivrv) and ivrv>=params["A"]["ivrv"]) and (abs(ag_z)<0.5):
        rec={"playbook":"A","estrutura":"Iron Condor / Short Strangle protegido","DTE":"7–15d","tamanho_cap": "0.5–0.8% risco",
             "targets":"theta 40–60% do prêmio","stops":"RV10 > 1.2× ou Dealer→NegGamma"}
        rationale.append("PosGamma + TR + IVRV alto + distância do Flip → vender vol.")
    elif (dealer=="NegGamma") and (wy_last.get("phase") in ["ACC","DST"]) and (wy_last.get("spring") or wy_last.get("ut") or wy_last.get("sweep_dn") or wy_last.get("sweep_up")) and (np.isfinite(ag_z) and abs(ag_z)>=params["B"]["ag_z"]):
        rec={"playbook":"B","estrutura":"Vertical (call/put) OU spot; straddle se IV%ile ≤ 35","DTE":"14–35d (spreads)","tamanho_cap":"0.5–0.8%",
             "targets":"1.0–1.5R e trailing 3 barras","stops":"fechamento de volta no range"}
        rationale.append("NegGamma + gatilho Wyckoff + agressão forte → direcional/compra de vol.")
    dtn,_ = days_to_opex(daily.index[-1])
    if rec is None and np.isfinite(dtn) and params["C"]["days_pre"][0] <= dtn <= params["C"]["days_pre"][1] and dealer=="PosGamma":
        rec={"playbook":"C","estrutura":"Butterfly no strike de maior OI","DTE":"D-1..3","tamanho_cap":"0.3–0.5%",
             "targets":"50–70% do valor máximo","stops":"sair antes do leilão"}
        rationale.append("Pré-OpEx com PosGamma + cluster de OI → borboleta (pin).")
    if rec is None and np.isfinite(dtn) and (-params["D"]["days_post"][1] <= -dtn <= -params["D"]["days_post"][0]) and dealer=="NegGamma":
        rec={"playbook":"D","estrutura":"Straddle/strangle leve OU spot com stop curto","DTE":"7–15d","tamanho_cap":"0.5%",
             "targets":"sair se moveu 1σ sem follow-through","stops":"perde momentum"}
        rationale.append("Pós-OpEx com NegGamma/instável → procurar rompimentos.")
    if rec is None:
        rec={"playbook":"—","estrutura":"n/d","DTE":"n/d","tamanho_cap":"n/d","targets":"n/d","stops":"n/d"}
        rationale.append("Sem condição forte de playbook agora.")
    rec["rationale"]="; ".join(rationale); rec["ivrv"]=_num(ivrv); rec["iv_pct"]=_num(iv_pct,'n/d','{:.0f}%')
    rec["dist_flip_atr"]=_num(dist_flip_atr)
    return rec

def final_synthesis(r: dict):
    tkr = r["ticker"]; parts = r["joint"]["parts"]
    wy = r["wy_last"]; ga = r["ga"]; ana = r["analog"]; ml = r["ml"]; walls = r["walls"]
    p_ml   = float(ml.get("prob_up_ml", np.nan))
    p_ana  = float(ana.get("prob_up", np.nan))
    p_dl   = float((r.get("dl") or {}).get("prob_up_dl", np.nan))
    p_up   = _prob_blend(float(r["joint"]["score"]), p_ml, p_ana, p_dl)
    p_dn   = (1.0 - p_up) if np.isfinite(p_up) else np.nan
    lo, hi, em_pct = _blended_move_band(r["daily"], ga, ana, days=10)
    bias = "Alta" if p_up>0.55 else ("Baixa" if p_up<0.45 else "Neutra")
    dealer = parts.get("dealer_regime", "n/d")
    flip   = parts.get("flip", None)
    ivrv   = ga.get("IVRV", np.nan)
    motivos = []
    reg = wy.get("regime", "n/d")
    if reg=="Markup": motivos.append("Wyckoff em **Markup**")
    elif reg=="Markdown": motivos.append("Wyckoff em **Markdown**")
    else: motivos.append("Wyckoff em **TR/Transição**")
    motivos.append(f"Dealer **{dealer}**" + (" (tende a range)" if dealer=="PosGamma" else " (tende a impulso)"))
    if np.isfinite(p_ml):  motivos.append(f"ML ↑ {p_ml:,.0%}")
    if np.isfinite(p_ana): motivos.append(f"Analógicos ↑ {p_ana:,.0%}")
    if np.isfinite(ivrv):
        if ivrv>=1.15: motivos.append(f"IV>RV ({ivrv:.2f}) — prêmio elevado")
        elif ivrv<=0.85: motivos.append(f"IV<RV ({ivrv:.2f}) — prêmio baixo")
    rec = playbook_signal(r, DEFAULT_PARAMS)
    texto = (
        f"**{tkr} — Súmula (10d)**\n"
        f"- Prob. **Alta**: **{(p_up if np.isfinite(p_up) else float('nan')):,.0%}** | **Baixa**: **{(p_dn if np.isfinite(p_dn) else float('nan')):,.0%}** "
        f"→ Viés: **{bias}**\n"
        f"- Faixa provável (±max EM): **{_num(lo)}/{_num(hi)}** (±{_num(em_pct,'n/d','{:.2f}')}%)\n"
        f"- **Playbook sugerido:** {rec['playbook']} — {rec['estrutura']} (DTE {rec['DTE']}, risco {rec['tamanho_cap']}).\n"
        f"- **Motivos:** " + "; ".join(motivos) + ".\n"
        f"- Níveis de opções: Flip {_num(flip)}, Put Wall {_num(walls.get('put_wall', np.nan))}, Call Wall {_num(walls.get('call_wall', np.nan))}.\n"
        f"- Racional do playbook: {rec['rationale']}"
    )
    return texto, p_up, em_pct, lo, hi, rec.get("playbook","—")

# ==========================
# Gráficos
# ==========================
def plot_asset_panel(
    daily: pd.DataFrame,
    ga: dict,
    gex_df: pd.DataFrame,
    walls: dict,
    title: str,
    flip_band_atr_mult: float = 0.5,   # largura da “zona flip” em ATR
):
    hist, proj = forecast_band_ar_garch(daily, ga, horizon=10, last_n=200)

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        specs=[[{"secondary_y": False}], [{"secondary_y": True}]]
    )

    idx = daily.index
    close = daily["close"].astype(float)
    fig.add_trace(go.Scatter(x=idx, y=close, mode="lines", name="Close",
                             line=dict(width=2, color=JP_COLORS["blue"])), row=1, col=1)

    if "ma200" in daily.columns and daily["ma200"].notna().any():
        fig.add_trace(go.Scatter(x=idx, y=daily["ma200"], mode="lines", name="MA200",
                                 line=dict(width=1.5, color=JP_COLORS["grey"], dash="dot")),
                      row=1, col=1)

    # Banda AR/GARCH
    t_up = go.Scatter(x=proj.index, y=proj["up1"], mode="lines",
                      line=dict(width=0.5, color="rgba(0,0,0,0)"),
                      showlegend=False, hoverinfo="skip")
    t_dn = go.Scatter(x=proj.index, y=proj["dn1"], mode="lines",
                      fill="tonexty", fillcolor="rgba(200,163,91,0.18)",
                      line=dict(width=0.5, color="rgba(0,0,0,0)"),
                      name="Faixa 1σ (AR/GARCH)")
    t_mean = go.Scatter(x=proj.index, y=proj["mean"], mode="lines+markers",
                        name="Caminho médio", line=dict(width=2, color=JP_COLORS["gold"]))
    fig.add_trace(t_up,   row=1, col=1)
    fig.add_trace(t_dn,   row=1, col=1)
    fig.add_trace(t_mean, row=1, col=1)

    # ---------------------------
    # PAREDES / REGIÕES (subplot 1)
    # ---------------------------
    spot = float(close.iloc[-1])
    atr  = float(daily["atr14"].iloc[-1]) if ("atr14" in daily.columns and np.isfinite(daily["atr14"].iloc[-1])) else np.nan

    put_w   = float(walls.get("put_wall", np.nan))
    call_w  = float(walls.get("call_wall", np.nan))
    flip    = float(walls.get("gamma_flip", np.nan))
    iv_maxk = float(walls.get("iv_max_strike", np.nan))
    iv_mink = float(walls.get("iv_min_strike", np.nan))

    # Faixa entre Put↔Call walls
    if np.isfinite(put_w) and np.isfinite(call_w) and (put_w != call_w):
        y0, y1 = (put_w, call_w) if put_w < call_w else (call_w, put_w)
        fig.add_hrect(
            y0=y0, y1=y1, line_width=0,
            fillcolor="rgba(83,182,126,0.10)", row=1, col=1,
            annotation_text="Range (Put↔Call walls)", annotation_position="inside top left",
            annotation_font_color=JP_COLORS["green"]
        )

    # Zona Flip (± k * ATR)
    if np.isfinite(flip) and np.isfinite(atr) and atr > 0:
        half = flip_band_atr_mult * atr
        fig.add_hrect(
            y0=flip - half, y1=flip + half, line_width=0,
            fillcolor="rgba(0,209,209,0.12)", row=1, col=1,
            annotation_text=f"Zona Flip (±{flip_band_atr_mult:.1f} ATR)",
            annotation_position="inside top right",
            annotation_font_color="#00D1D1"
        )

    # Linhas de referência
    def _hline(yv, color, dash, txt, pos="top right"):
        if np.isfinite(yv):
            fig.add_hline(y=float(yv), line_color=color, line_width=1.6, line_dash=dash,
                          annotation_text=txt, annotation_position=pos,
                          annotation_font_color=color, row=1, col=1)

    _hline(call_w,  JP_COLORS["red"],   "dot",     "Call Wall")
    _hline(put_w,   JP_COLORS["green"], "dot",     "Put Wall")
    _hline(flip,    "#00D1D1",          "dash",    "Gamma Flip")
    _hline(iv_maxk, JP_COLORS["gold"],  "dashdot", "IV Máx", pos="bottom right")
    _hline(iv_mink, JP_COLORS["blue"],  "dashdot", "IV Mín", pos="bottom right")

    # ---------------------------
    # Volume / Agressão (subplot 2)
    # ---------------------------
    vol = daily.get("vol", pd.Series(index=daily.index, data=np.nan))
    fig.add_trace(go.Bar(x=idx, y=vol, name="Volume"), row=2, col=1, secondary_y=False)

    ag = daily.get("aggr", pd.Series(index=daily.index, data=np.nan))
    if np.isfinite(ag).sum() > 0:
        fig.add_trace(go.Scatter(x=idx, y=ag, mode="lines", name="Agressão",
                                 line=dict(width=1.6)), row=2, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Volume", row=2, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Agressão", row=2, col=1, secondary_y=True)
    else:
        fig.update_yaxes(title_text="Volume", row=2, col=1)

    # Estilo
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=JP_COLORS["bg"], plot_bgcolor=JP_COLORS["panel"],
        font=dict(color=JP_COLORS["txt"]), title=title,
        margin=dict(l=40, r=20, t=60, b=40), legend=dict(bgcolor="rgba(0,0,0,0)")
    )
    fig.update_xaxes(showgrid=True, gridcolor=JP_COLORS["grid"], row=1, col=1)
    fig.update_yaxes(showgrid=True, gridcolor=JP_COLORS["grid"], row=1, col=1)
    fig.update_xaxes(showgrid=True, gridcolor=JP_COLORS["grid"], row=2, col=1)
    fig.update_yaxes(showgrid=True, gridcolor=JP_COLORS["grid"], row=2, col=1)

    return fig


def build_gex_table(gex_df: pd.DataFrame, unit_lbl_from_calc: str | None = None):
    """
    Monta a TABELA de exposição em gamma por strike.
    Retorna apenas Strike, Call GEX (+) e Put GEX (−), ordenada por Strike.
    """
    if gex_df is None or gex_df.empty:
        cols = ["Strike", "Call GEX (+)", "Put GEX (−)"]
        return pd.DataFrame(columns=cols), "R$ / 1% move"

    unit_lbl = unit_lbl_from_calc or "R$ / 1% move"

    T = (
        gex_df[["strike", "call_gex", "put_gex"]]
        .rename(columns={
            "strike": "Strike",
            "call_gex": "Call GEX (+)",
            "put_gex": "Put GEX (−)",
        })
        .sort_values("Strike")
        .reset_index(drop=True)
    )
    return T, unit_lbl

def plot_gex_histogram_net(gex_df: pd.DataFrame, walls: dict, daily: pd.DataFrame, target_bars: int = 160):
    """
    Histograma HORIZONTAL **apenas do NET** (CALL − PUT) por strike.
    - X: NET GEX (assinado)
    - Y: Strike
    Faz binning só quando há strikes demais (>= target_bars).
    """
    if gex_df is None or gex_df.empty:
        return go.Figure()

    T = gex_df.sort_values("strike").reset_index(drop=True).copy()
    y_strike = T["strike"].astype(float).values
    x_net    = T["total_gex"].astype(float).values

    def _nice_step(raw):
        if not np.isfinite(raw) or raw <= 0: return 0.1
        m = 10 ** np.floor(np.log10(raw))
        for mult in (1, 2, 5, 10):
            if raw <= mult * m: return mult * m
        return 10 * m

    unique_strikes = np.unique(y_strike)
    lo, hi = float(np.nanmin(y_strike)), float(np.nanmax(y_strike))
    rng = hi - lo

    # Binning só quando necessário
    if len(unique_strikes) > max(1, int(target_bars)) and rng > 0:
        step = _nice_step(rng / target_bars)
        bins = np.floor((y_strike - lo) / step).astype(int)
        centers = lo + (bins + 0.5) * step
        Tb = pd.DataFrame({"y": centers, "n": x_net})
        Tbin = Tb.groupby("y", as_index=False).sum().sort_values("y")
        y = Tbin["y"].values
        x_net = Tbin["n"].values
        bar_width = step * 0.85
    else:
        y = y_strike
        if len(unique_strikes) >= 2:
            diffs = np.diff(unique_strikes)
            step = float(np.nanpercentile(diffs[diffs > 0], 25)) if (diffs > 0).any() else float(diffs[0])
        else:
            step = max(0.01, (hi or 1.0) * 0.01)
        bar_width = step * 0.90

    units_lbl = (walls or {}).get("_gex_units", "R$ / 1% move")

    # Cores por sinal (opcional, melhora leitura)
    colors = [JP_COLORS["gold"] if v >= 0 else JP_COLORS["blue"] for v in x_net]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=y, x=x_net, width=bar_width, orientation="h",
        name="NET (CALL − PUT)",
        marker=dict(color=colors, line=dict(width=0.2, color="rgba(255,255,255,0.25)"))
    ))

    # Linhas de referência (horizontais, por strike)
    spot = float(daily["close"].iloc[-1])
    fig.add_hline(y=spot, line_dash="dash", line_width=1.2, line_color="#AAAAAA",
                  annotation_text="Spot", annotation_position="top right")

    def _hline(yv, color, dash, txt, pos="top left"):
        if np.isfinite(yv):
            fig.add_hline(y=float(yv), line_color=color, line_width=1.4, line_dash=dash,
                          annotation_text=txt, annotation_position=pos, annotation_font_color=color)

    _hline(walls.get("call_wall",  np.nan), JP_COLORS["red"],   "dot",     "Call Wall")
    _hline(walls.get("put_wall",   np.nan), JP_COLORS["green"], "dot",     "Put Wall")
    _hline(walls.get("gamma_flip", np.nan), "#00D1D1",          "dash",    "Gamma Flip")
    _hline(walls.get("iv_max_strike", np.nan), JP_COLORS["gold"], "dashdot", "IV Máx")
    _hline(walls.get("iv_min_strike", np.nan), JP_COLORS["blue"], "dashdot", "IV Mín")

    # Zero do NET (vertical em x=0)
    fig.add_vline(x=0, line_dash="dot", line_width=1, line_color="#999999")

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=JP_COLORS["bg"], plot_bgcolor=JP_COLORS["panel"],
        font=dict(color=JP_COLORS["txt"]),
        title="GEX — NET (CALL − PUT) por Strike (1% move) — barras horizontais",
        margin=dict(l=60, r=40, t=60, b=50),
        bargap=0.10, showlegend=True
    )
    fig.update_xaxes(title_text=f"Gamma Exposure ({units_lbl})", showgrid=True, gridcolor=JP_COLORS["grid"])
    fig.update_yaxes(title_text="Strike", showgrid=True, gridcolor=JP_COLORS["grid"])
    return fig

def latest_by_strike(chain: pd.DataFrame) -> pd.DataFrame:
    t = chain.copy()
    t["date_n"] = pd.to_datetime(t.get("date"), errors="coerce")
    t["strike"] = pd.to_numeric(t.get("strike"), errors="coerce")
    t = t[(t["strike"] > 0)]
    t = t.sort_values(["type", "strike", "date_n"]).drop_duplicates(subset=["type", "strike"], keep="last")
    return t

# ==========================
# Pipeline por ativo
# ==========================
def analyze_asset_file(path: str):
    base = os.path.basename(str(path))
    ticker = os.path.splitext(base)[0].upper()

    try:
        for _ in range(3):
            try:
                with open(path, "rb") as f:
                    buf = io.BytesIO(f.read())
                break
            except PermissionError:
                time.sleep(0.5)
        chain, daily = read_chain_and_daily_auto(buf, sheet_name=0)
    except Exception as e:
        return {"ticker": ticker, "error": f"leitura falhou: {e}"}
        # --- NOVO: normalização automática (centavos → reais) ---
    chain, daily, _scale_info = _autoscale_cents(chain, daily)

    

    if daily.empty or daily["close"].dropna().empty:
        return {"ticker": ticker, "error": "sem dados de preço"}

    spot = float(daily["close"].iloc[-1])

    ch = chain.copy()
    for col in ["gamma", "delta", "vega", "theta", "C. Abertos", "prob_itm", "prob_otm"]:
        if col not in ch.columns:
            ch[col] = 0.0

    last_opt_day = _pick_last_options_day(ch, min_strikes=5)
    if last_opt_day is not None:
        dnorm = pd.to_datetime(ch.get("date"), errors="coerce").dt.normalize()
        ch_last = ch.loc[dnorm == last_opt_day].copy()
        if ch_last.empty:
            ch_last = latest_by_strike(ch)
    else:
        ch_last = latest_by_strike(ch)

    df_gex, walls, _ = compute_exposures(
        ch_last, spot,
        contract_size_override=None,
        per_1pct=True,
        out_unit="auto"
    )

    params = calibrate_wyckoff_params(daily)
    wy = wyckoff_signals(
        daily,
        lookback_tr=20,
        tr_ratio_thresh=params["tr_ratio_thresh"],
        wick_z=params["wick_z"],
        eff_z_thresh=params["eff_z_thresh"],
    )
    wy = smooth_wyckoff_blocks(wy, min_len=3)
    wy["fase_pt"] = phase_label_pt(wy)
    wy = wyckoff_pro_enhance(daily, wy)
    wy_last = wy.iloc[-1]

    ret = np.log(daily["close"]).diff()
    ga = garch_arma_signals(ret, ch_last)
    flow_df, micro_diag = spot_player_features(daily, wy)
    expo_ts = exposures_time_series(chain, daily, max_days=400)

    S = build_state_features(daily, wy, flow_df, expo_ts=expo_ts)
    analog = find_analogs(S, daily["close"], horizon=10, k=60)
    ml = ml_directional_prob(S, daily["close"], horizon=10, min_train=150)
    drifts = drift_family(daily["close"])
    lin = linear_return_mu(S, daily["close"], horizon=10)
    dl = lstm_directional_prob(daily["close"], horizon=10, lookback=30)
    rl = kelly_dir(ga.get("mu_1d", np.nan), ga.get("sigma_1d", np.nan))
    joint = joint_directional_view_hybrid(
        wy_last, walls, spot, df_gex, ga, analog, ml, drifts=drifts, lin=lin, dl=dl, rl=rl
    )

    return {
        "ticker": ticker,
        "spot": spot,
        "wy": wy,
        "wy_last": wy_last,
        "ga": ga,
        "gex_df": df_gex,
        "walls": walls,
        "analog": analog,
        "ml": ml,
        "drifts": drifts,
        "lin": lin,
        "dl": dl,
        "rl": rl,
        "joint": joint,
        "dealer": joint["parts"]["dealer_regime"],
        "flip": joint["parts"]["flip"],
        "micro_diag": micro_diag,
        "daily": daily,
        "expo_ts": expo_ts,
        "flow": micro_diag,
        "chain": ch,
    }

# ==========================
# Sanidade & agregação
# ==========================
def validate_data(chain: pd.DataFrame, daily: pd.DataFrame):
    msgs=[]
    if "close" not in daily.columns or daily["close"].dropna().empty: msgs.append("Preço: série 'close' vazia.")
    if (daily.index.duplicated().any()): msgs.append("Datas duplicadas no diário.")
    if "iv" in chain.columns:
        iv=_normalize_iv_series(chain["iv"])
        if (iv<0).any(): msgs.append("IV negativa detectada.")
    if "oi" in chain.columns and (pd.to_numeric(chain["oi"], errors="coerce")<0).any(): msgs.append("OI negativo.")
    if chain is None or chain.empty:
        msgs.append("Cadeia de opções vazia — verifique colunas de Strike/OI/Gamma/IV ou a planilha (aba) correta.")
    return msgs

def aggregate_market(results: list, weights: dict):
    """
    Agrega leituras de vários ativos ponderando por peso.
    - Direcional agora vem de PROBABILIDADE ponderada (não só do score).
    - μ_1d = soma_i w_i * μ_i
    - σ_1d ≈ sqrt( soma_i (w_i^2 * σ_i^2) )   [sem covariâncias]
    """
    # Pesos apenas dos tickers presentes e normalizados
    present = {r["ticker"]: float(weights.get(r["ticker"], 0.0)) for r in results if "error" not in r}
    w = pd.Series(present, dtype=float)
    w = w[w > 0]
    w = (w / w.sum()) if w.sum() > 0 else w

    # Acumuladores
    prob_num, prob_den = 0.0, 0.0         # para prob. agregada ponderada
    score_raw = 0.0                        # score ponderado (apenas informativo / fallback)
    wy_counts = {}                         # fase dominante por peso
    pos_gamma_w = 0.0                      # “massa” ponderada em PosGamma
    mu_w, var_w = 0.0, 0.0                 # μ e variância agregados

    for r in results:
        if "error" in r: 
            continue
        wt = float(w.get(r["ticker"], 0.0))
        if wt <= 0: 
            continue

        # --- Probabilidade por ativo (blend de fontes) ---
        ml_p  = float((r.get("ml") or {}).get("prob_up_ml", np.nan))
        ana_p = float((r.get("analog") or {}).get("prob_up", np.nan))
        dl_p  = float((r.get("dl") or {}).get("prob_up_dl", np.nan))
        jsc   = float((r.get("joint") or {}).get("score", np.nan))

        # p_asset: usa o blend existente; fallback para ensemble→prob se necessário
        p_asset = _prob_blend(jsc, ml_p, ana_p, dl_p)  # pode retornar NaN
        if not np.isfinite(p_asset) and np.isfinite(jsc):
            p_asset = 0.5 + 0.5 * float(np.clip(jsc, -1.0, 1.0))

        if np.isfinite(p_asset):
            prob_num += wt * p_asset
            prob_den += wt

        if np.isfinite(jsc):
            score_raw += wt * jsc

        # --- Wyckoff (fase dominante) ponderado ---
        ph_pt = str(r.get("wy_last", {}).get("fase_pt", "Transição"))
        wy_counts[ph_pt] = wy_counts.get(ph_pt, 0.0) + wt

        # --- Dealer agregado (peso da massa PosGamma) ---
        if r.get("dealer", "PosGamma") == "PosGamma":
            pos_gamma_w += wt

        # --- AR/GARCH agregados ---
        mu_i = float((r.get("ga") or {}).get("mu_1d", 0.0))
        sig_i = float((r.get("ga") or {}).get("sigma_1d", np.nan))
        if np.isfinite(mu_i):
            mu_w += wt * mu_i
        if np.isfinite(sig_i):
            var_w += (wt ** 2) * (sig_i ** 2)

    # Probabilidade agregada
    prob_up = float(prob_num / prob_den) if prob_den > 0 else np.nan
    # Score derivado da probabilidade (em [-1,1]); fallback para score_raw se prob for NaN
    score = float(2.0 * prob_up - 1.0) if np.isfinite(prob_up) else float(score_raw)

    # Sinal & confiança
    if np.isfinite(prob_up):
        signal = "UP" if prob_up > 0.55 else ("DOWN" if prob_up < 0.45 else "NEUTRAL")
        confidence = int(np.clip(40 + 60 * abs(score), 0, 100))
    else:
        signal = "UP" if score_raw > 0.10 else ("DOWN" if score_raw < -0.10 else "NEUTRAL")
        confidence = int(np.clip(45 + 55 * abs(score_raw), 0, 100))

    dom_phase = max(wy_counts, key=lambda k: wy_counts[k]) if wy_counts else "n/d"
    dealer_regime = "PosGamma" if pos_gamma_w >= 0.5 else "NegGamma"

    return {
        "signal": signal,
        "score": round(float(score), 3),            # agora coerente com a prob agregada
        "confidence": confidence,
        "prob_up": (None if not np.isfinite(prob_up) else float(prob_up)),
        "dominant_phase": dom_phase,
        "dealer": dealer_regime,
        "prob_posgamma": round(float(pos_gamma_w), 3),  # “massa” em PosGamma
        "mu_1d": round(float(mu_w), 5),
        "sigma_1d": round(float(np.sqrt(var_w)) if var_w > 0 else np.nan, 5),
    }

def previsao_texto(r: dict):
    sig = r["joint"]["signal"]
    dealer = r.get("dealer","PosGamma")
    reg = r.get("wy_last",{}).get("regime","TR")
    if sig=="UP": return "Subindo"
    if sig=="DOWN": return "Descendo"
    if dealer=="PosGamma" and reg in ["TR","Transição"]: return "Consolidando"
    return "Neutro"

# ==========================
# Pesos
# ==========================
EMBEDDED_WEIGHTS = {"VALE3":22.21,"ITUB4":16.882,"PETR4":12.708,"PETR3":8.6,"BBDC4":7.7,"SBSP3":7.5,"ELET3":7.2,"B3SA3":5.9,"ITSA4":5.9,"BBAS3":5.4}
def _normalize_weights(w): ser=pd.Series(w,dtype=float); ser= ser/ser.sum() if ser.sum()!=0 else ser; return ser.to_dict()
def load_weights(weights_csv: str | None, use_embedded=True):
    if weights_csv and os.path.exists(weights_csv):
        df=pd.read_csv(weights_csv); w={str(r["ticker"]).upper(): float(r["weight"]) for _,r in df.iterrows()}; return _normalize_weights(w)
    if use_embedded: return _normalize_weights(EMBEDDED_WEIGHTS.copy())
    return {}

# ==========================
# STREAMLIT — UI integrada
# ==========================
def _mtime_key(path: str) -> float:
    try: return os.path.getmtime(path)
    except Exception: return time.time()

def run_streamlit_multi(data_dir="data", weights_csv=None, use_embedded_weights=True):
    # PRIMEIRO comando do Streamlit no ciclo de execução:
    st.set_page_config(page_title="Leitura Integrada (IBOV)", layout="wide")
    st.title("Leitura Integrada — Wyckoff + GEX + AR/GARCH + Analógicos + ML")

    st.sidebar.header("Entrada")
    st.sidebar.write("Coloque **.xlsx** no diretório configurado, cada um com OHLC/Volume/Agressão e, se houver, cadeia de opções.")
    weights = load_weights(weights_csv, use_embedded=use_embedded_weights)
    if not weights:
        st.sidebar.error("Sem pesos carregados. Use os embutidos ou forneça CSV (ticker,weight)."); return
    st.sidebar.success("Pesos carregados.")

    paths = sorted(glob.glob(os.path.join(data_dir, "*.xlsx")))
    if not paths:
        st.warning(f"Nenhum .xlsx encontrado em `{data_dir}`."); st.stop()
    st.sidebar.write(f"Arquivos encontrados: {len(paths)}")

    @st.cache_data(show_spinner=False)
     def _cached_analyze(path, mtime): 
         return analyze_asset_file(path)

    with st.spinner("Processando ativos…"):
        args=[(p,_mtime_key(p)) for p in paths]
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(6,len(paths))) as ex:
            results=list(ex.map(lambda t: _cached_analyze(*t), args))

    errs=[r for r in results if "error" in r]
    if errs:
        with st.expander("Falhas de leitura", expanded=False):
            for e in errs: st.write(f"**{e['ticker']}** → {e['error']}")

    ok=[r for r in results if "error" not in r]
    if not ok: st.stop()

    with st.expander("Sanity checks (dados)"):
        for r in ok:
            msgs=validate_data(r.get("chain", pd.DataFrame()), r["daily"])
            if msgs: st.write(f"**{r['ticker']}**: " + " | ".join(msgs))

    agg=aggregate_market(ok, weights)

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div style="padding:12px;border-radius:12px;background:{'#2ecc71' if agg['signal']=='UP' else '#e74c3c' if agg['signal']=='DOWN' else '#7f8c8d'};color:white">
        <b>Direcional</b><br><span style="font-size:26px">{agg['signal']}</span><br>
        <small>conf. {agg['confidence']}% • score {agg['score']}</small>
        </div>""", unsafe_allow_html=True)
    with c2: st.metric("Wyckoff (fase dominante)", agg["dominant_phase"])
    with c3: st.metric("Dealer (agregado)", "PosGamma" if agg["dealer"]=="PosGamma" else "NegGamma",
                       delta=f"prob PosΓ {int(100*agg['prob_posgamma'])}%")
    with c4: st.metric("AR(1) μ esperado (1D)", f"{agg['mu_1d']*100:.2f}%")
    with c5: st.metric("Vol 1D prevista (GARCH)", f"{agg['sigma_1d']*100:.2f}%")

    st.markdown("---"); st.markdown("---")

    tab_sumula, tab_paineis, tab_gex, tab_tabela, tab_ml = st.tabs(
    ["Súmula por Ativo", "Gráficos (Preço+Fluxo+Paredes)", "GEX (Tabela/Histograma)", "Tabela Geral (prévia 10d)", "ML / Analíticos"]
    )

    with tab_sumula:
        tkr = st.selectbox("Ativo", options=[r["ticker"] for r in ok], key="sum_tkr")
        rs  = next(x for x in ok if x["ticker"]==tkr)
        txt, p_up, em_pct, lo, hi, pb = final_synthesis(rs)
        st.markdown(txt)

    with tab_paineis:
        tkr=st.selectbox("Ativo", options=[r["ticker"] for r in ok], key="panel_tkr")
        rs=next(x for x in ok if x["ticker"]==tkr)
        colA, colB = st.columns([3,2])
        with colA:
            fig=plot_asset_panel(rs["daily"], rs["ga"], rs["gex_df"], rs["walls"],
                                 f"{tkr} — Preço, Projeção e Banda 1σ (AR/GARCH)")
            st.plotly_chart(fig, use_container_width=True)
        with colB:
            st.markdown(f"**Resumo executivo**")
            parts = rs["joint"]["parts"]
            st.write(f"- **Ensemble:** {rs['joint']['signal']} (score {rs['joint']['score']:+.3f}, conf. {rs['joint']['confidence']}%)")
            st.write(f"- **Dealer:** {parts.get('dealer_regime','n/d')} • **Flip:** {_num(parts.get('flip',np.nan))}")
            em_abs, em_pct = expected_move_days(rs["ga"], float(rs["spot"]), days=10)
            st.write(f"- **EM 10d (GARCH):** ±{_num(em_pct,'n/d','{:.2f}')}% (≈ ±{_num(em_abs,'n/d','{:.2f}')} pts)")
            st.write(f"- **ML (logística):** prob. alta {rs.get('ml',{}).get('prob_up_ml', float('nan')):.2%}")
            st.write(f"- **Deep LSTM:** prob. alta {rs.get('dl',{}).get('prob_up_dl', float('nan')):.2%}")
            st.write(f"- Contribuições → Wy: {parts['wyckoff_dir']:+.2f} | GEX: {parts['gex_dir']:+.2f} | Ana: {parts['analog_dir']:+.2f} | ML: {parts['ml_dir']:+.2f}")

    with tab_gex:
        tkr = st.selectbox("Ativo", options=[r["ticker"] for r in ok], key="gex_tkr")
        rs  = next(x for x in ok if x["ticker"] == tkr)

        # Histograma GEX (NOVO: exibido de fato)
        fig_gex = plot_gex_histogram_net(rs["gex_df"], rs["walls"], rs["daily"], target_bars=9999)
        st.plotly_chart(fig_gex, use_container_width=True)

        dbg = st.toggle("Debug GEX deste ativo", value=False)
        if dbg:
            ch = rs.get("chain", pd.DataFrame())
            st.write({
                "linhas_chain": len(ch),
                "colunas_chain": list(ch.columns) if not ch.empty else [],
                "strikes_unicos": int(pd.to_numeric(ch.get("strike"), errors="coerce").dropna().astype(float).nunique()) if ("strike" in ch) else 0,
                "tipos": sorted(ch["type"].dropna().unique().tolist()) if ("type" in ch and not ch.empty) else []
            })
            st.dataframe(ch.head(20))

        with st.expander("Depurar GEX deste ativo", expanded=False):
            ch = rs["chain"].copy()
            s  = pd.to_numeric(ch.get("strike"), errors="coerce")
            oi = pd.to_numeric(ch.get("oi"), errors="coerce")
            gm = pd.to_numeric(ch.get("gamma"), errors="coerce")
            iv = _normalize_iv_series(ch.get("iv", pd.Series(np.nan, index=ch.index)))
            st.write({
                "linhas_chain": len(ch),
                "strikes>0": int((s > 0).sum()),
                "C": int((ch.get("type","").astype(str).str.upper().str[0] == "C").sum()),
                "P": int((ch.get("type","").astype(str).str.upper().str[0] == "P").sum()),
                "oi>0": int((oi > 0).sum()),
                "gamma!=0": int(gm.fillna(0).abs().gt(0).sum()),
                "iv>0": int((iv > 0).sum()),
                "cov_gamma%": rs["walls"].get("cov_gamma"),
                "cov_oi%": rs["walls"].get("cov_oi"),
                "cov_iv%": rs["walls"].get("cov_iv"),
            })
            st.caption("Se strikes>0 for 0 → não achou a coluna Strike/Exercício. "
                       "Se oi>0 for 0 → a coluna de OI não foi encontrada (ou veio tudo 0).")

        tbl, unit_lbl = build_gex_table(rs["gex_df"], rs["walls"].get("_gex_units"))
        if tbl.empty:
            st.info("Sem GEX calculado para este ativo.")
        else:
            st.markdown(f"**Unidade:** {unit_lbl}")
            st.dataframe(
                tbl.style.format({
                    "Strike": "{:.2f}",
                    "Call GEX (+)": "{:+,.0f}",
                    "Put GEX (−)": "{:+,.0f}",
                }),
                use_container_width=True,
                height=520,
            )
            st.caption(
                "Referência de layout: na planilha de origem **CALLs** ficam à esquerda da coluna **Strike** "
                "e **PUTs** à direita."
            )

    with tab_tabela:
        rows=[]
        for r in ok:
            _, p, em, lo_, hi_, pb_ = final_synthesis(r)
            walls=r.get("walls",{}) or {}; put_wall=walls.get("put_wall",np.nan); call_wall=walls.get("call_wall",np.nan); flip=r.get("flip",np.nan)
            rows.append({
                "ticker": r["ticker"],
                "score": r["joint"]["score"],
                "sinal": r["joint"]["signal"],
                "previsao_10d": previsao_texto(r),
                "prob_up": p,
                "faixa_±%": em,
                "low": lo_, "high": hi_,
                "dealer": r.get("dealer","n/d"),
                "flip": _num(flip,'n/d','{:.2f}'),
                "put_wall": _num(put_wall,'n/d','{:.2f}'),
                "call_wall": _num(call_wall,'n/d','{:.2f}'),
                "fase": r.get("wy_last",{}).get("fase_pt","n/d"),
                "regime": r.get("wy_last",{}).get("regime","n/d"),
                "playbook": pb_,
            })
        df_sum = pd.DataFrame(rows)
        st.dataframe(
            df_sum.sort_values(["previsao_10d","score"], ascending=[True, False]).style.format({
                "prob_up":"{:.0%}", "faixa_±%":"{:.2f}", "low":"{:.2f}", "high":"{:.2f}", "score":"{:+.3f}"
            }),
            use_container_width=True, height=520
        )

    with tab_ml:
        rows=[]
        for r in ok:
            ml=r.get("ml",{}) or {}; dl=r.get("dl",{}) or {}; dr=r.get("drifts",{}) or {}; lin=r.get("lin",{}) or {}
            rows.append({
                "ticker": r["ticker"],
                "prob_up_ml": ml.get("prob_up_ml", np.nan),
                "auc_cv": ml.get("auc_cv", np.nan),
                "prob_up_dl": dl.get("prob_up_dl", np.nan),
                "mu_lin": lin.get("mu_lin", np.nan),
                "mu_auto": dr.get("mu_auto", np.nan),
                "mu_sarima": dr.get("mu_sarima", np.nan)
            })
        df_ml=pd.DataFrame(rows).sort_values("prob_up_ml", ascending=False)
        st.dataframe(
            df_ml.style.format({
                "prob_up_ml":"{:.2%}", "auc_cv":"{:.2f}",
                "prob_up_dl":"{:.2%}", "mu_lin":"{:+.2%}",
                "mu_auto":"{:+.2%}", "mu_sarima":"{:+.2%}"
            }),
            use_container_width=True, height=380
        )
        with st.expander("Como ler estes estudos", expanded=False):
            st.markdown("""
- **ML (logística)**: usa features (Wyckoff, fluxo, técnicos, dealer) → probabilidade de fechar **acima** em 10d.  
- **AUC (CV)**: qualidade média do classificador em validação temporal (0.5=aleatório; 0.7+ já é útil).  
- **Deep LSTM**: rede simples de série temporal com janelas de retornos → probabilidade direcional (10d).  
- **Linear (features→ret)**: regressão do retorno log em 10d nas features do dia → *μ* esperado.  
- **ARIMA/SARIMA**: *drifts* na série de retornos, independentes das features.  
- **EM (GARCH)**: banda de movimento esperado (1σ) usada nos gráficos.
            """.strip())
        st.markdown("---")
        tkr = st.selectbox("Detalhar ativo", options=[r["ticker"] for r in ok], key="ml_detail_tkr")
        rs  = next(x for x in ok if x["ticker"]==tkr)
        ml, ana, dr, lin = rs.get("ml",{}), rs.get("analog",{}), rs.get("drifts",{}), rs.get("lin",{})
        dl = rs.get("dl",{})
        prob_ml = ml.get("prob_up_ml", np.nan)
        auc     = ml.get("auc_cv", np.nan)
        pos_feats, neg_feats = _top_features(ml.get("coef", {}), k=5)
        p_ana   = ana.get("prob_up", np.nan)
        r20, r80 = ana.get("ret_q20", np.nan), ana.get("ret_q80", np.nan)
        em_pct_ana = ana.get("em_pct_ana", np.nan)
        mu_lin  = lin.get("mu_lin", np.nan)
        mu_auto = dr.get("mu_auto", np.nan)
        mu_sar  = dr.get("mu_sarima", np.nan)
        p_dl    = dl.get("prob_up_dl", np.nan)
        st.subheader(f"{tkr} — O que os estudos estão dizendo")
        bullet = []
        if np.isfinite(prob_ml):
            bullet.append(f"**ML (logística):** prob. de alta {prob_ml:.2%}" + (f" • AUC {auc:.2f}" if np.isfinite(auc) else ""))
        if np.isfinite(p_ana):
            qtxt = (f"ret q20 {r20:+.2%}, q80 {r80:+.2%}" if np.isfinite(r20) and np.isfinite(r80) else "quantis indisponíveis")
            bullet.append(f"**Analógicos:** prob. de alta {p_ana:.2%} • EM(10d) ≈ ±{em_pct_ana:.2f}% • {qtxt}")
        if np.isfinite(mu_lin):
            bullet.append(f"**Linear (features→ret):** μ ≈ {mu_lin:+.2%}")
        if np.isfinite(mu_auto) or np.isfinite(mu_sar):
            comp = mu_auto if np.isfinite(mu_auto) else mu_sar
            bullet.append(f"**ARIMA/SARIMA (drift):** μ ≈ {comp:+.2%}")
        if np.isfinite(p_dl):
            bullet.append(f"**DL (LSTM):** prob. de alta {p_dl:.2%}")
        st.markdown("- " + "\n- ".join(bullet) if bullet else "_Sem amostra suficiente para ML/Analíticos._")
        if pos_feats or neg_feats:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Principais drivers (↑):**")
                if pos_feats:
                    st.write("\n".join([f"- {k}: {v:+.3f}" for k,v in pos_feats]))
                else:
                    st.write("_n/d_")
            with col2:
                st.markdown("**Principais penalizadores (↓):**")
                if neg_feats:
                    st.write("\n".join([f"- {k}: {v:+.3f}" for k, v in neg_feats]))
                else:
                    st.write("_n/d_")
        st.caption("Obs.: coeficientes positivos empurram a probabilidade para cima; negativos, para baixo (no contexto do modelo logístico).")

# Execução direta
if __name__ == "__main__":
    run_streamlit_multi(data_dir="data", weights_csv=None, use_embedded_weights=True)

