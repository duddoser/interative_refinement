import pandas as pd

def make_excel(input_csv: str, output_xlsx: str):
    """
    Формирует Excel с:
      - 1-я строка: описания колонок
      - 2-я строка: технические имена колонок
      - Данные с 3-й строки
      - Подсветка каждый 7-й строки (Excel: 3,10,17,...)
    """
    df = pd.read_csv(input_csv)

    descriptions = {
        'Size':       'Размер матрицы (m×n)',
        'Interval':   'Интервал сингулярных значений [a;b]',
        'CondNum':    'Число обусловленности\n(max(sv)/min(sv))',
        'NoiseLevel': 'Уровень шума\n(макс. изменение в U и V)',
        'Iter':       'Число итераций\nOgita–Aishima',
        'Rec_l1':     'L1-норма невязки\nA - U·S·Vᵀ',
        'Rec_l2':     'Frobenius-норма\nA - U·S·Vᵀ',
        'Time_ms':    'Время выполнения\n(мс)',
        'U_err':      'Ошибка U:\n||U_est - U_true||₂',
        'S_err':      'Ошибка S:\n||S_est - S_true||₂',
        'V_err':      'Ошибка V:\n||V_est - V_true||₂'
    }

    header_desc = [descriptions.get(col, col) for col in df.columns]
    header_key  = list(df.columns)

    with pd.ExcelWriter(output_xlsx, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Results', startrow=2, index=False, header=False)
        wb = writer.book
        ws = writer.sheets['Results']

        # Форматы
        fmt_desc = wb.add_format({
            'bold': True, 'text_wrap': True,
            'align': 'center', 'valign': 'vcenter',
            'fg_color': '#DCE6F1', 'border': 1
        })
        fmt_key = wb.add_format({
            'bold': True,
            'align': 'center', 'valign': 'vcenter',
            'fg_color': '#EBF1DE', 'border': 1
        })
        fmt_highlight = wb.add_format({'bg_color': '#F2F2F2'})

        ws.set_row(0, 40)
        ws.set_row(1, 25)
        for col_idx, text in enumerate(header_desc):
            ws.write(0, col_idx, text, fmt_desc)
        for col_idx, text in enumerate(header_key):
            ws.write(1, col_idx, text, fmt_key)

        # Ширина колонок
        for i, col in enumerate(header_key):
            max_len = max(
                df[col].astype(str).map(len).max(),
                len(header_desc[i]), len(col)
            ) + 2
            ws.set_column(i, i, max_len)

        ws.freeze_panes(2, 0)

        n = len(df)
        for row in range(2, 2 + n, 7):
            ws.set_row(row, None, fmt_highlight)

    print(f'Готово: {output_xlsx}')

if __name__ == '__main__':
    make_excel(
        input_csv='results.csv',
        output_xlsx='results.xlsx'
    )
