# Bibliotecas
from shiny.express import ui, render, input
from shiny.ui import page_navbar
from shinyswatch import theme
import pandas as pd
import plotnine as p9
import mizani as mi
from functools import partial

# Funções
def preparar_dados(parquet, y):
    df_tmp = pd.read_parquet(parquet).reset_index(names = "data")
    periodo_prev = df_tmp.filter(["data", "Previsão"]).dropna().data
    df_tmp.iloc[periodo_prev.index.min() - 1] = df_tmp.iloc[periodo_prev.index.min() - 1].ffill()
    return df_tmp.melt(id_vars = ["data", "Intervalo Inferior", "Intervalo Superior"])

def gerar_grafico(df, y, n, unidade, linha_zero = True):
    dt = input.periodo()
    df_tmp = (
        df
        .drop(
            labels = ["Intervalo Inferior", "Intervalo Superior"],
            axis = "columns"
            )
        .query("variable == @y and data >= @dt")
        .dropna()
        )
    df_prev = (
        df
        .query("variable == 'Previsão'")
        .dropna()
        .head(n) # workaround para problema de groups do plotnine
        )
    def plotar_zero():
        if linha_zero:
            return p9.geom_hline(yintercept = 0, linetype = "dashed")
        else:
            return None
    plt = (
        p9.ggplot(df_tmp) +
        p9.aes(x = "data", y = "value", color = "variable") +
        plotar_zero() +
        p9.geom_ribbon(
            data = df_prev,
            mapping = p9.aes(ymin = "Intervalo Inferior", ymax = "Intervalo Superior"),
            alpha = 0.25 if input.ic() else 0, 
            fill = "blue", 
            color = "none"
            ) +
        p9.geom_line() +
        p9.geom_line(data = df_prev, group = 1) +
        p9.scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
        p9.scale_y_continuous(breaks = mi.breaks.breaks_extended(6)) +
        p9.scale_color_manual(values = {y: "black", "Previsão": "blue"}) +
        p9.labs(color = "", x = "", y = unidade) +
        p9.theme(
            panel_grid_minor = p9.element_blank(),
            legend_position = "bottom"
            )
    )
    return plt

# Dados
df_ipca = preparar_dados("previsao/ipca.parquet", "IPCA")
df_cambio = preparar_dados("previsao/cambio.parquet", "Câmbio")
df_pib = preparar_dados("previsao/pib.parquet", "PIB")
df_selic = preparar_dados("previsao/selic.parquet", "Selic")
df_divpib = preparar_dados("previsao/divpib.parquet", "DIV/PIB")
df_endivfam = preparar_dados("previsao/endivfam.parquet", "Endiv. Famílias")

# Layout
ui.page_opts(
    title = ui.span(
        ui.img(
            src = "https://aluno.analisemacro.com.br/download/59787/?tmstv=1712933415",
            height = 30,
            style = "padding-right:10px;"
            )
    ),
    window_title = "Painel de Previsões",
    page_fn = partial(page_navbar, fillable = True),
    theme = theme.minty
)

with ui.nav_panel("Painel de Previsões"):  
    with ui.layout_sidebar():
        
        # Inputs
        with ui.sidebar(width = 225):
            ui.input_slider(
                id = "h",
                label = "Horizonte de previsão:",
                min = 1,
                max = 12,
                value = 12,
                step = 1, 
                width = "100%",
                post = "m"
                )
            ui.input_date(
                id = "periodo",
                label = "Início do gráfico:",
                value = pd.to_datetime("today") - pd.offsets.YearBegin(7),
                min = "2004-01-01",
                max = df_selic.data.max(),
                language = "pt-BR",
                width = "100%"
                )
            ui.input_checkbox(
                id = "ic",
                label = "Intervalo de confiança",
                value = True
            )
        
        # Outputs
        with ui.layout_column_wrap():
            
            with ui.card():
                ui.card_header(ui.strong("Inflação (IPCA)"))
                @render.plot
                def ipca():
                    plt = gerar_grafico(df_ipca, "IPCA", input.h() + 1, "Var. %")
                    return plt

            with ui.card():
                ui.card_header(ui.strong("Taxa de Câmbio (BRL/USD)"))
                @render.plot
                def cambio():
                    plt = gerar_grafico(df_cambio, "Câmbio", input.h() + 1, "R\$/US\$", False)
                    return plt

        with ui.layout_column_wrap():
            
            with ui.card():
                ui.card_header(ui.strong("Atividade Econômica (PIB)"))
                @render.plot
                def pib():
                    plt = gerar_grafico(df_pib, "PIB", (input.h()//3) + 1, "Var. % anual")
                    return plt

            with ui.card():
                ui.card_header(ui.strong("Taxa de Juros (SELIC)"))
                @render.plot
                def selic():
                    plt = gerar_grafico(df_selic, "Selic", input.h() + 1, "% a.a.", False)
                    return plt

            with ui.card():
                ui.card_header(ui.strong("DLSP/PIB"))
                @render.plot
                def selic():
                    plt = gerar_grafico(df_divpib, "DIV/PIB", input.h() + 1, "%", False)
                    return plt

            with ui.card():
                ui.card_header(ui.strong("Endividamento das famílias exceto crédito habitacional (RNDBF)"))
                @render.plot
                def selic():
                    plt = gerar_grafico(df_endivfam, "Endiv. Famílias", input.h() + 1, "% da renda disp.", False)
                    return plt
