import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import pandas as pd
import numpy_financial as npf
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import logging 
df = px.data.gapminder().query("year == 2007")
unit_df = pd.DataFrame()

def recalc_dataframe(
    users_acquisited: int = 1000,
    customers: int = 10,
    churn_rate: float = 0.5,
    AVP: float = 50,
    APC: int = 2,
    TMS: float = 10000,
    COGS: float = 0,
    COGS1: float = 0,
    fixed_cost_server: float = 20000,
    fixed_cost_domain: float = 10000,
    periods: int = 24,
    tax: float = 0.15,
    discount_rate: float = 0.15,
    required_investments: float = 0
) -> pd.DataFrame:
    
    df = pd.DataFrame(index=range(periods))
    

    df["users_acquisited"] = users_acquisited
    df["customers"] = customers
    df["churn_rate"] = churn_rate
    df["acquisition_rate"] = 1.0
    df["user_churn"] = 0.0
    df["total_users"] = df["users_acquisited"]
    df["C1"] = df["customers"] / df["total_users"]
    df["AVP"] = AVP
    df["APC"] = APC
    df["TMS"] = TMS
    df["COGS"] = COGS
    df["COGS1"] = COGS1
    df["fixed_cost_server"] = fixed_cost_server
    df["fixed_cost_domain"] = fixed_cost_domain
    df["fixed_costs"] = fixed_cost_domain + fixed_cost_server
    df["IDT"] = tax


    for i in range(1, len(df)):
        df.loc[i, "user_churn"] = int(df.loc[i, "churn_rate"] * df.loc[i-1, "total_users"])#type:ignore
        df.loc[i, "total_users"] = (
            df.loc[i, "users_acquisited"] + #type:ignore
            df.loc[i-1, "total_users"] - 
            df.loc[i, "user_churn"]) #type:ignore
        df.loc[i, "acquisition_rate"] = df.loc[i, "users_acquisited"] / df.loc[i, "total_users"]#type:ignore
    

    df["customers"] = df["total_users"] * df["C1"]
    

    df["ARPC"] = df["AVP"] * df["APC"]
    df["ARPU"] = df["AVP"] * df["APC"] * df["C1"]
    df["UAC"] = df["TMS"] / df["users_acquisited"]
    df["CAC"] = df["UAC"] / df["C1"]
    df["CLTV"] = (df["AVP"] - df["COGS"]) * df["APC"] - df["COGS1"]
    df["LTV"] = df["CLTV"] * df["C1"]
    df["CM"] = df["LTV"] - df["UAC"]
    df["CCM"] = df["CLTV"] - df["CAC"]
    df["revenue"] = df["ARPU"] * df["total_users"]
    df["gross_profit"] = df["LTV"] * df["total_users"]
    df["margin"] = df["CM"] * df["total_users"]
    df["margin_rate"] = df["margin"] / df["revenue"]
    df["EBIDTA"] = df["margin"] - df["fixed_costs"]
    df["net_profit"] = df["EBIDTA"] * (1 - df["IDT"])
    

    df["accumulative_profit"] = df["net_profit"].cumsum()
    

    df["ROMI"] = df["margin"] / df["TMS"]
    

    df["cash_flow"] = df["net_profit"]
    df.loc[0, "ballance"] = -required_investments + df.loc[0, "net_profit"]#type:ignore
    for i in range(1, len(df)):
        df.loc[i, "ballance"] = df.loc[i-1, "ballance"] + df.loc[i, "net_profit"]#type:ignore
    

    cf = df["cash_flow"].to_numpy()
    npv_values = np.array([np.sum(cf[:i+1] / (1 + discount_rate) ** np.arange(i+1)) for i in range(len(cf))])
    df["NPV"] = npv_values
    
    def calculate_irr(cash_flows, required_investments):
        """
        Правильный расчет IRR с инвестициями в периоде 0
        """
        # Создаем полный денежный поток: инвестиции в периоде 0, затем операционные потоки
        full_cash_flow = np.array([-required_investments] + cash_flows.tolist())
        
        # Проверяем условия для расчета IRR
        has_negative = np.any(full_cash_flow < 0)
        has_positive = np.any(full_cash_flow > 0)
        
        if not (has_negative and has_positive):
            return np.nan  # Нельзя рассчитать IRR
        
        try:
            irr_value = npf.irr(full_cash_flow)
            
            # Проверяем, что IRR действительное число
            if np.iscomplex(irr_value):
                return np.nan
            if np.isnan(irr_value):
                return np.nan
                
            return irr_value
            
        except Exception as e:
            logging.error(f"Error calculating IRR: {e}")
            return np.nan

    df["IRR"] = calculate_irr(df["cash_flow"].to_numpy(), required_investments)

    df["ROI"] = np.where(required_investments > 0, 
                        df["net_profit"] / required_investments, 
                        np.nan)
    
    return df

app = dash.Dash(__name__)

input_style = {
    'width': '250px',
    'padding': '6px',
    'margin': '5px 0 15px 0',
    'border-radius': '5px',
    'border': '1px solid #ccc',
    'font-size': '14px'
}

button_style = {
    'backgroundColor':'#007BFF',
    'color':'white',
    'padding':'10px 20px',
    'border':'none',
    'border-radius':'5px',
    'cursor':'pointer',
    'font-size':'16px',
    'margin-top':'10px'
}

label_style = {
    'font-weight':'bold',
    'margin-top':'10px',
    'display':'block',
    'color':'#333'
}

app.layout = html.Div([
    html.H2("Параметры расчёта", style={'textAlign':'center', 'color':'#222', 'margin-bottom':'20px'}),

    html.Div([
        html.Label("Количество привлекаемых пользователей в месяц", style=label_style),
        dcc.Input(id="users_input", type="number", value=11500, style=input_style),

        html.Label("Количество клиентов в месяц", style=label_style),
        dcc.Input(id="customers_input", type="number", value=1520, style=input_style),

        html.Label("Коэфициент оттока", style=label_style),
        dcc.Input(id="churn_rate_input", type="number", value=0.2, style=input_style),

        html.Label("Средний чек (AVP)", style=label_style),
        dcc.Input(id="avp_input", type="number", value=10, style=input_style),

        html.Label("Кол-во сделок на клиента (APC)", style=label_style),
        dcc.Input(id="apc_input", type="number", value=3, style=input_style),

        html.Label("Маркетинговый бюджет (TMS)", style=label_style),
        dcc.Input(id="tms_input", type="number", value=8000, style=input_style),

        html.Label("Себестоимость сделки (COGS)", style=label_style),
        dcc.Input(id="cogs_input", type="number", value=0, style=input_style),

        html.Label("Себестоимость 1 сделки (COGS1)", style=label_style),
        dcc.Input(id="cogs1_input", type="number", value=0, style=input_style),

        html.Label("Стоимость аренды сервера", style=label_style),
        dcc.Input(id="server_cost_input", type="number", value=10000, style=input_style),

        html.Label("Стоимость домена", style=label_style),
        dcc.Input(id="domain_cost_input", type="number", value=5000, style=input_style),

        html.Label("Кол-во месяцев расчета", style=label_style),
        dcc.Input(id="month_count_input", type="number", value=36, style=input_style),

        html.Label("Налоговая ставка (доходы-расходы)", style=label_style),
        dcc.Input(id="tax_rate_input", type="number", value=0.15, style=input_style),

        html.Label("Ставка дисконтирования (ставка ЦБ)", style=label_style),
        dcc.Input(id="discount_rate_input", type="number", value=0.18, style=input_style),

        html.Label("Первичный инвестиционный капитал", style=label_style),
        dcc.Input(id="required_investments_input", type="number", value=20000, style=input_style),

        html.Br(),
        html.Button("Пересчитать", id="run_button", n_clicks=0, style=button_style)
    ], style={'display':'flex', 'flexDirection':'column', 'alignItems':'flex-start'}),

    html.Hr(),
    html.Div(id="result_output")
], style={'padding':'20px', 'font-family':'Arial, sans-serif'})

@app.callback(
    Output("result_output", "children"),
    Input("run_button", "n_clicks"),
    State("users_input", "value"),
    State("customers_input", "value"),
    State("churn_rate_input", "value"),
    State("avp_input", "value"),
    State("apc_input", "value"),
    State("tms_input", "value"),
    State("cogs_input", "value"),
    State("cogs1_input", "value"),
    State("server_cost_input", "value"),
    State("domain_cost_input", "value"),
    State("month_count_input", "value"),
    State("tax_rate_input", "value"),
    State("discount_rate_input", "value"),
    State("required_investments_input", "value"),
)
def run_model(
    n_clicks, users, customers, churn, avp, apc, tms,
    cogs, cogs1, server_cost, domain_cost, months,
    tax_rate, discount_rate, investments
):
    if n_clicks == 0:
        return "Введите параметры и нажмите кнопку"

    df = recalc_dataframe(
        users_acquisited=users,
        customers=customers,
        churn_rate=churn,
        AVP=avp,
        APC=apc,
        TMS=tms,
        COGS=cogs,
        COGS1=cogs1,
        fixed_cost_server=server_cost,
        fixed_cost_domain=domain_cost,
        periods=months,
        tax=tax_rate,
        discount_rate=discount_rate,
        required_investments=investments
    )

    irr = df["IRR"].iloc[0]
    irr_str = f"{irr:.2%}" if isinstance(irr, (int, float)) and not pd.isna(irr) else "Not profitable"


    colors = px.colors.qualitative.Pastel


    net_profit_plot = px.line(df, y="net_profit", title="Чистая прибыль", markers=True, line_shape="spline", color_discrete_sequence=[colors[0]])
    accumulative_profit_plot = px.area(df, y="accumulative_profit", title="Накопленная прибыль", color_discrete_sequence=[colors[1]])
    ballance_plot = px.line(df, y="ballance", title="Баланс компании", markers=True, color_discrete_sequence=[colors[2]])
    EBITDA_plot = px.bar(df, y="EBIDTA", title="EBITDA по периодам", color_discrete_sequence=[colors[3]])
    revenue_plot = px.bar(df, y="revenue", title="Выручка", color_discrete_sequence=[colors[4]])
    gross_profit_plot = px.bar(df, y="gross_profit", title="Валовая прибыль", color_discrete_sequence=[colors[5]])
    user_count_plot = px.bar(df, y="total_users", title="Общее количество пользователей", color_discrete_sequence=[colors[6]])
    margin_plot = px.line(df, y="margin", title="Маржа", markers=True, color_discrete_sequence=[colors[7]])


    cm_value = df["CM"].iloc[-1]
    cm_color = "green" if cm_value >= 0 else "red"
    cm_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=cm_value,
        title={'text': "Contribution Margin (CM)", 'font': {'size': 14}},
        gauge={
            'axis': {'range': [df["CM"].min(), df["CM"].max()], 'tickfont': {'size': 10}},
            'bar': {'color': cm_color},
            'bgcolor': "#f0f0f0",
            'borderwidth': 1,
            'bordercolor': "#ccc",
            'steps': [
                {'range': [df["CM"].min(), 0], 'color': '#ffe6e6'},
                {'range': [0, df["CM"].max()], 'color': '#e6ffe6'}
            ]
        },
        number={'font': {'size': 16}}
    ))
    cm_gauge.update_layout(width=250, height=250, margin=dict(l=20, r=20, t=40, b=20))


    ltv_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=df["LTV"].iloc[-1],
        title={'text': "LTV", 'font': {'size': 14}},
        gauge={
            'axis': {'range': [0, df["LTV"].max()*1.2], 'tickfont': {'size': 10}},
            'bar': {'color': "#4d79ff"},
            'bgcolor': "#f0f0f0",
            'borderwidth': 1,
            'bordercolor': "#ccc"
        },
        number={'font': {'size': 16}}
    ))
    ltv_gauge.update_layout(width=250, height=250, margin=dict(l=20, r=20, t=40, b=20))


    uac_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=df["UAC"].iloc[-1],
        title={'text': "UAC", 'font': {'size': 14}},
        gauge={
            'axis': {'range': [0, df["UAC"].max()*1.2], 'tickfont': {'size': 10}},
            'bar': {'color': "#ffa64d"},
            'bgcolor': "#f0f0f0",
            'borderwidth': 1,
            'bordercolor': "#ccc"
        },
        number={'font': {'size': 16}}
    ))
    uac_gauge.update_layout(width=250, height=250, margin=dict(l=20, r=20, t=40, b=20))

    return html.Div([
        html.H4("Финансовые показатели:"),
        html.P(f"NPV: {int(df['NPV'].iloc[0]):,} ₽"),
        html.P(f"IRR: {irr_str}"),
        html.P(f"ROI: {df['ROI'].iloc[0] if isinstance(df['ROI'].iloc[0], str) else f'{df['ROI'].iloc[0]:.2%}'}"),

        html.Hr(),
        html.H5("Графики финансовых показателей:"),
        html.Div([
            dcc.Graph(figure=net_profit_plot),
            dcc.Graph(figure=accumulative_profit_plot),
            dcc.Graph(figure=ballance_plot),
            dcc.Graph(figure=EBITDA_plot),
        ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(2, 1fr)', 'gap': '20px'}),

        html.Div([
            dcc.Graph(figure=revenue_plot),
            dcc.Graph(figure=gross_profit_plot),
            dcc.Graph(figure=user_count_plot),
            dcc.Graph(figure=margin_plot),
        ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(2, 1fr)', 'gap': '20px'}),

        html.Hr(),
        html.H5("Ключевые показатели (Gauge):"),
        html.Div([
            dcc.Graph(figure=cm_gauge),
            dcc.Graph(figure=ltv_gauge),
            dcc.Graph(figure=uac_gauge),
        ], style={'display': 'flex', 'justifyContent': 'space-around'}),

        html.Hr(),
        html.H5("Полная таблица данных:"),
        html.Pre(df.to_string(index=False))
    ])
if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)