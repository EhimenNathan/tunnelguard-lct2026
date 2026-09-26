"""Build the final presentation from the official template.

Mandatory slides 7-11 are filled in place (exact template design kept). Solution slides are built in the template's
own visual language (purple background with partner logos, pink title pill, white rounded cards with blush outline,
template palette) and follow the pitch structure requested in the case: problem -> idea -> algorithm -> demo ->
results -> what worked / what did not. Numbers come from report_numbers.json (make_report.py) and the figures from
make_deck_figures.py / make_video.py. Run from the scratchpad directory that holds report_numbers.json and logos/.
"""
import copy
import json
import os

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Cm, Pt

TEMPLATE = os.environ.get('DECK_TEMPLATE', os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'ЛЦТ2026 Шаблон презентации.pptx'))
SOL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.environ.get('DECK_OUT', os.path.join(SOL, 'presentation'))
FIG = os.path.join(SOL, 'docs', 'figures')
DFIG = os.path.join(FIG, 'deck')
LOGO = 'logos/Image_2.png'
FONT = 'Montserrat'
MONO = 'Consolas'
R = json.load(open('report_numbers.json'))
DEMO = json.load(open(os.path.join(FIG, 'demo_log.json')))
# deployed model (v3 + ego-motion test): held-out recording, 58 of 60 in-gauge frames STOP, first STOP from the video run
R.update(hold_detected=58, hold_stop=58, hold_dist=round(DEMO['holdout_first_stop'][1], 1))
os.makedirs(OUT_DIR, exist_ok=True)
prs = Presentation(TEMPLATE)
S = prs.slides

INK = RGBColor(0x1C, 0x1D, 0x22)
MUTED = RGBColor(0x5E, 0x60, 0x70)
PURPLE = RGBColor(0x52, 0x09, 0x77)
PINK = RGBColor(0xFF, 0x00, 0x53)
BLUSH = RGBColor(0xFF, 0xD6, 0xE3)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
SOFT = RGBColor(0xEA, 0xDC, 0xF3)

fp_loro = 0.31
ABLATION_NOTES = json.load(open('ablation_notes.json', encoding='utf-8')) if os.path.exists('ablation_notes.json') else []
fp_rules_loro = 1.31
syn = R['synth']


def recall(shape, d0):
    rows = [r for r in syn if r[0] == shape and r[1] == d0]
    return 100 * rows[0][4] if rows else float('nan')


# ============================================================== helpers: template slides
def set_text(shape, lines, size=None, bold=None, color=None, font=None):
    """Replace text keeping the first run's formatting; lines = list of str (one paragraph each)."""
    tf = shape.text_frame
    p0 = tf.paragraphs[0]
    rpr = copy.deepcopy(p0.runs[0]._r.find(qn('a:rPr'))) if p0.runs else None
    ppr = copy.deepcopy(p0._p.find(qn('a:pPr')))
    for p in tf.paragraphs[1:]:
        p._p.getparent().remove(p._p)
    for r in list(p0.runs):
        r._r.getparent().remove(r._r)
    for br in p0._p.findall(qn('a:br')):
        p0._p.remove(br)
    for i, line in enumerate(lines):
        p = p0 if i == 0 else tf.add_paragraph()
        if i > 0 and ppr is not None:
            p._p.insert(0, copy.deepcopy(ppr))
        run = p.add_run()
        run.text = line
        if rpr is not None:
            run._r.insert(0, copy.deepcopy(rpr))
        if size:
            run.font.size = Pt(size)
        if bold is not None:
            run.font.bold = bold
        if color is not None:
            run.font.color.rgb = color
        if font:
            run.font.name = font


def by_text(slide, startswith):
    for sh in slide.shapes:
        if sh.has_text_frame and sh.text_frame.text.strip().startswith(startswith):
            return sh
    raise KeyError(startswith)


def placeholder(slide, idx):
    for sh in slide.placeholders:
        if sh.placeholder_format.idx == idx:
            return sh
    raise KeyError(idx)


def clone(src_no):
    src = S[src_no - 1]
    new = prs.slides.add_slide(src.slide_layout)
    for shp in list(new.shapes):
        shp._element.getparent().remove(shp._element)
    for shp in src.shapes:
        new.shapes._spTree.insert_element_before(copy.deepcopy(shp._element), 'p:extLst')
    return new


def text_width_cm(s, size_pt, bold=True, widen=1.10):
    from PIL import ImageFont
    f = ImageFont.truetype(os.path.join(os.environ.get('WINDIR', r'C:\Windows'), 'Fonts',
                                        'segoeuib.ttf' if bold else 'segoeui.ttf'), 200)
    return f.getlength(s) / 200 * size_pt / 72 * 2.54 * widen


def title_pill(slide, text):
    """Template title: white bold text on the pink pill (pill width follows the measured text)."""
    assert text_width_cm(text, 20) < 12.6, f'title too long for the pill next to the logos: {text}'
    set_text(placeholder(slide, 0), [text], font=FONT, bold=True, size=20)
    for sh in slide.shapes:
        if sh.shape_type == 1 and abs(sh.top - Cm(0.9)) < Cm(0.3) and sh.height < Cm(2.0) and sh.left < Cm(1.5):
            sh.width = Cm(text_width_cm(text, 20) + 1.6)


def base_slide(title):
    """Clone of template slide 16 reduced to background + title pill + slide number."""
    sl = clone(16)
    for sh in list(sl.shapes):
        keep = (sh.is_placeholder and sh.placeholder_format.idx in (0, 4)) or (
            sh.shape_type == 1 and abs(sh.top - Cm(0.9)) < Cm(0.3) and sh.height < Cm(2.0) and sh.left < Cm(1.5))
        if not keep:
            sh._element.getparent().remove(sh._element)
    title_pill(sl, title)
    return sl


# ============================================================== helpers: components
def text(slide, x, y, w, h, paras, size=12, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         font=FONT, spacing=1.1, space_after=4, inset=0.1):
    """paras: list of str or (str, dict(size,color,bold,font)) or list of runs [(str, dict), ...]."""
    tb = slide.shapes.add_textbox(Cm(x), Cm(y), Cm(w), Cm(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for side in ('left', 'right', 'top', 'bottom'):
        setattr(tf, f'margin_{side}', Cm(inset))
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        p.space_after = Pt(space_after)
        runs = para if isinstance(para, list) else [para]
        for run in runs:
            t, st = (run, {}) if isinstance(run, str) else run
            r = p.add_run()
            r.text = t
            r.font.size = Pt(st.get('size', size))
            r.font.bold = st.get('bold', bold)
            r.font.color.rgb = st.get('color', color)
            r.font.name = st.get('font', font)
    return tb


def card(slide, x, y, w, h, fill=WHITE, line=BLUSH, radius=0.05):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Cm(x), Cm(y), Cm(w), Cm(h))
    sh.adjustments[0] = radius
    sh.fill.solid(); sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line; sh.line.width = Pt(1.25)
    sh.shadow.inherit = False
    sh.text_frame.text = ''
    return sh


def picture_fit(slide, path, x, y, w, h, align='center'):
    from PIL import Image
    iw, ih = Image.open(path).size
    scale = min(w / iw, h / ih)
    pw, ph = iw * scale, ih * scale
    px = x + (w - pw) / 2 if align == 'center' else x
    return slide.shapes.add_picture(path, Cm(px), Cm(y + (h - ph) / 2), Cm(pw), Cm(ph))


def figure_card(slide, path, x, y, w, h, pad=0.35):
    card(slide, x, y, w, h, radius=0.035)
    return picture_fit(slide, path, x + pad, y + pad, w - 2 * pad, h - 2 * pad)


def kpi(slide, x, y, w, h, value, label, vcolor=PINK, vsize=26, lsize=10.5):
    card(slide, x, y, w, h, radius=0.08)
    text(slide, x + 0.35, y + 0.2, w - 0.7, h * 0.52, [value], size=vsize, bold=True, color=vcolor,
         anchor=MSO_ANCHOR.BOTTOM, spacing=0.9, space_after=0)
    text(slide, x + 0.35, y + h * 0.55, w - 0.7, h * 0.43, [label], size=lsize, color=INK, spacing=1.0, space_after=0)


def caption_bar(slide, x, y, w, h, paras, size=12):
    card(slide, x, y, w, h, fill=WHITE, line=None, radius=0.2)
    text(slide, x + 0.4, y, w - 0.8, h, paras, size=size, anchor=MSO_ANCHOR.MIDDLE, space_after=2)


def on_purple(slide, x, y, w, h, paras, size=12, **kw):
    return text(slide, x, y, w, h, paras, size=size, color=WHITE, **kw)


def numbered_cards(slide, items, top=3.4, height=13.9, gap=0.5, left=1.0, right=32.9):
    n = len(items)
    w = (right - left - gap * (n - 1)) / n
    for i, (title, body) in enumerate(items):
        x = left + i * (w + gap)
        card(slide, x, top, w, height, radius=0.05)
        bsize = 15 if n <= 3 else 13.5
        text(slide, x + 0.5, top + 0.45, 3.0, 1.6, [f'{i + 1:02d}'], size=28, bold=True, color=PINK)
        text(slide, x + 0.5, top + 2.2, w - 1.0, 2.4, [title], size=17 if n <= 3 else 15, bold=True, color=PURPLE,
             spacing=1.0)
        text(slide, x + 0.5, top + 4.6, w - 1.0, height - 5.0, body if isinstance(body, list) else [body], size=bsize,
             color=INK, spacing=1.15, space_after=10)


# ============================================================== mandatory slides 7-11 (template design kept)
s7 = S[6]
set_text(placeholder(s7, 0), ['[НАЗВАНИЕ КОМАНДЫ]'])
set_text(placeholder(s7, 12), ['TunnelGuard — обнаружение посторонних объектов в габарите беспилотного поезда метро '
                               'по данным 3D-лидара'])
placeholder(s7, 11).insert_picture(LOGO)

s8 = S[7]
set_text(placeholder(s8, 0), ['[Название команды]'])
set_text(by_text(s8, 'В чем суть'), [
    'Лидар сам восстанавливает путь и сечение тоннеля до горизонта видимости и в каждом кадре отвечает: есть ли '
    'в габарите поезда что-то постороннее, на каком расстоянии и насколько далеко путь проверен свободным.'], size=11, color=INK)
set_text(by_text(s8, 'Что делает'), [
    'Мы не распознаём классы объектов — мы моделируем «нормальный тоннель» и физику лидара, а ML лишь уточняет '
    f'решение. На новой линии (запечатанный тест) — 1.8 ложных остановки на км, в 6 раз меньше прежнего; реальный '
    f'человек в габарите — STOP на {R["hold_dist"]:.1f} м в {R["hold_detected"]} из {R["hold_inside"]} кадров.'], size=11, color=INK)

set_text(placeholder(S[8], 0), ['КОМАНДА'])

s10 = S[9]
set_text(placeholder(s10, 0), ['О КОМАНДЕ'])
set_text(by_text(s10, 'Что вас вдохновило'), [
    'Ошибка стоит дорого в обе стороны: пропуск — угроза людям, ложная тревога — остановка линии. '
    'Хотелось решить задачу не «чёрным ящиком», а объяснимой физикой тоннеля.'], size=12, color=INK)
set_text(by_text(s10, 'Расскажите о самых интересных'), [
    '23 ГБ записей при 3 ГБ свободного диска: написали потоковый разбор SQLite прямо из архива. Кривые, стрелки, '
    'станции ломали геометрию — каждый класс ошибок закрыли физическим принципом. Запись с людьми держали как отложенный тест.'],
    size=12, color=INK)

s11 = S[10]
set_text(placeholder(s11, 38), [
    'ROS 2 Humble + Docker; вход — любой PointCloud2, выход — ObstacleStatus, vision_msgs, маркеры RViz2.',
    'Путь из лидара в каждом кадре: рельсы (DP) → шаблон сечения → глобальный DP → Гаусс–Ньютон с клотоидой и σ(x).',
    'STOP — только при уверенном (2σ) нахождении в габарите; дальше 30 м — гибрид с ансамблем ML.',
    f'Итог: 1.8 ложных остановки на км новой линии (было 10.3); человек в габарите — {R["hold_detected"]}/'
    f'{R["hold_inside"]} кадров, STOP на {R["hold_dist"]:.1f} м; 56–70 мс на кадр.'], size=13.5)
set_text(placeholder(s11, 42), [
    '«Второй машинист», который не устаёт: ранний STOP и расстояние до препятствия для системы торможения.',
    'Установка на новый состав без ручной настройки: авто-поиск топика и оси лидара, калибровка по рельсам.',
    'Объяснимость для сертификации: каждое решение трассируется до геометрии и неопределённости.',
    'Развитие: C++/CUDA-ядро, слияние с камерой, карта тоннеля, мониторинг инфраструктуры.'], size=13.5)

# ============================================================== 12. problem
sl = base_slide('ПРОБЛЕМА')
numbered_cards(sl, [
    ('Тоннель — не дорога', ['Кривые R ≈ 300–1000 м, уклоны и «горбы» станций, стрелки, платформы, колонны между путями.',
                             'Прямоугольник «перед поездом» либо тонет в ложных тревогах на кривой, либо смотрит мимо пути.']),
    ('На 200 м — единицы точек', ['Pandar128: 0.125° по вертикали — на 160 м шаг лучей 0.35 м, человек даёт ≈ 9 отражений.',
                                   'Стены видны под скользящим углом, дальняя геометрия пути неоднозначна.']),
    ('Ложная тревога = остановка линии', ['Большинство записей — пустой тоннель: размеченных препятствий почти нет.',
                                          'Нужен не только ответ «есть / нет», но и честная оценка: насколько далеко путь '
                                          'действительно проверен.']),
])

# ============================================================== 13. idea
sl = base_slide('ИДЕЯ')
numbered_cards(sl, [
    ('Путь из данных', 'Рельсы, сечение тоннеля, кривизна, уклон и крен восстанавливаются лидаром в каждом кадре — без карт и калибровки.'),
    ('Габарит из данных', 'Свободное пространство всех записей в координатах пути задаёт форму габарита: контактный рельс, платформы, стены, свод.'),
    ('Решение с σ', 'У каждой точки пути есть неопределённость σ(x). STOP — только если объект внутри габарита, сжатого на 2σ, и в пределах измеренной дальности.'),
    ('Физика + ML', 'Инварианты: объект не «сдвигает» путь, инфраструктура крепится к оболочке. Ансамбль ML на физических признаках отсеивает ложные кластеры дальше 30 м.'),
], height=13.9)

# ============================================================== 14. architecture
sl = base_slide('АРХИТЕКТУРА')
figure_card(sl, os.path.join(DFIG, 'architecture.png'), 1.0, 3.3, 31.9, 12.4)
caption_bar(sl, 1.0, 16.1, 30.0, 1.5, [[('Запуск: ', dict(bold=True, color=PURPLE)),
            ('docker build → docker run → ros2 bag play → /tunnel_guard/status.  ', {}),
            ('Одно ядро', dict(bold=True, color=PURPLE)),
            (' без зависимостей от ROS: узел, оффлайн-оценка evaluate_bag и 10 тестов pytest.', {})]], size=12)

# ============================================================== 15. geometry
sl = base_slide('ГЕОМЕТРИЯ ПУТИ')
figure_card(sl, os.path.join(FIG, 'geometry.png'), 1.0, 3.3, 21.4, 9.2)
card(sl, 22.9, 3.3, 10.0, 14.3, radius=0.05)
text(sl, 23.4, 3.7, 9.0, 13.5, [
    ('Как восстанавливается путь', dict(size=15, bold=True, color=PURPLE)),
    [('Рельсы 3–45 м: ', dict(bold=True)), ('гребни BEV + пара рельсов 1520 мм, глобальный DP Витерби.', {})],
    [('Шаблон сечения: ', dict(bold=True)), ('ближние точки в координатах пути — модель «нормального тоннеля».', {})],
    [('Глобальный DP до 300 м: ', dict(bold=True)), ('путь не может уйти вбок на метры — компактный объект его не изогнёт.', {})],
    [('Гаусс–Ньютон: ', dict(bold=True)), ('каждая точка ложится на шаблон; клотоидный априор, веса Тьюки.', {})],
    [('Инвариантность: ', dict(bold=True)), ('точки внутри габарита не участвуют в оценке пути.', {})],
    [('Калман по времени: ', dict(bold=True)), ('сливается только измеренное; экстраполяция растит σ.', {})],
], size=11, spacing=1.08, space_after=7)
card(sl, 1.0, 12.9, 21.4, 4.7, radius=0.06)
text(sl, 1.5, 13.15, 20.4, 4.3, [
    ('Проверено на синтетическом тоннеле с эталонной геометрией и лучевым моделированием', dict(bold=True, color=PURPLE, size=12)),
    'Человек на 160 м больше не смещает оценку пути (ошибка < 10 см). Видимость на кривой ограничена хордой √(8RW) — '
    'дальше неё путь физически закрыт стеной, и решение честно ограничивается этой дальностью.',
], size=11.5, spacing=1.1)

# ============================================================== 16. envelope
sl = base_slide('ГАБАРИТ ИЗ ДАННЫХ')
figure_card(sl, os.path.join(FIG, 'envelope.png'), 1.0, 3.3, 15.2, 14.3)
card(sl, 16.7, 3.3, 16.2, 14.3, radius=0.05)
text(sl, 17.3, 3.8, 15.0, 13.4, [
    ('Что измерено по всем записям', dict(size=15, bold=True, color=PURPLE)),
    'Контактный рельс: |l| = 1.22 м на высоте 0.2–0.45 м',
    'Край платформы: ≥ 1.37 м (h 0.6–1.4 м)',
    'Стены: ≥ 1.48 м; свод круглого тоннеля 1.18 м на высоте 3.43 м; потолок прямоугольного 3.8 м',
    'Объекты ниже 0.15 м над головкой рельса игнорируются',
    ('Зоны решения', dict(size=15, bold=True, color=PURPLE)),
    [('ВНУТРИ', dict(bold=True, color=PINK)), (' — внутри габарита, сжатого на 2σ → кандидат в STOP', {})],
    [('РЯДОМ', dict(bold=True, color=PURPLE)), (' — касание в пределах погрешности → CAUTION (информирует, не тормозит)', {})],
    'Габарит обнаружения лежит внутри свободного пространства всех тоннелей — поезда физически проходят через него.',
], size=12, spacing=1.1, space_after=8)

# ============================================================== 17. hybrid model
sl = base_slide('ФИЗИКА + ML')
figure_card(sl, os.path.join(DFIG, 'hybrid.png'), 1.0, 3.3, 31.9, 9.4)
cols = [
    ('Данные без утечки', '38 физических признаков кластера. Негативы — все кандидаты 5 пустых записей; позитивы — лучевое '
                          'моделирование объектов в реальных лучах. Запись с людьми не используется.'),
    ('Честная проверка', 'Обучение на 4 тоннелях — тест на 5-м (leave-one-tunnel-out), порог — вложенной CV. Сравнение '
                         'на уровне решений: ложные STOP-кадры и полнота.'),
    ('Две модели', 'v1: ⅓·LightGBM + ⅓·CatBoost + ⅓·PI-MLP (датасет 1). Финал: ½·LightGBM + ½·CatBoost + ЭГО-тест '
                   '(датасеты 1 + 2). До 30 м решают правила; вывод на numpy (JSON 0.3 МБ).'),
]
for i, (t, b) in enumerate(cols):
    x = 1.0 + i * 10.8
    card(sl, x, 13.1, 10.3, 4.5, radius=0.07)
    text(sl, x + 0.4, 13.3, 9.5, 4.2, [(t, dict(bold=True, color=PURPLE, size=13)), b], size=10.5, spacing=1.05, space_after=4)

# ============================================================== model stack
sl = base_slide('МОДЕЛЬ')
figure_card(sl, os.path.join(DFIG, 'model_stack.png'), 1.0, 3.3, 22.4, 10.4)
card(sl, 23.9, 3.3, 9.0, 14.3, radius=0.05)
text(sl, 24.4, 3.7, 8.0, 13.6, [
    ('Финальная модель', dict(size=15, bold=True, color=PURPLE)),
    ('Роутер двух ансамблей ½·LightGBM + ½·CatBoost + ЭГО-тест', dict(size=12.5, bold=True, color=PINK)),
    [('Роутер: ', dict(bold=True)), ('кандидат на полу → эксперт по людям и коробкам; низом ≥ 0.5 м → эксперт по '
                                      'парящим и висящим объектам. У каждого свой порог вне фолда.', {})],
    [('LightGBM ', dict(bold=True)), ('с ограничениями монотонности: больше точек в габарите (у эксперта «на полу» — и опоры на пол) — оценка '
                                      'не может упасть.', {})],
    [('CatBoost ', dict(bold=True)), ('— симметричные деревья, устойчивы к шуму.', {})],
    [('ЭГО-тест ', dict(bold=True)), ('без обучения: по лидарной одометрии объект обязан приближаться со скоростью '
                                      'поезда; «едущий» с поездом артефакт → только ВНИМАНИЕ.', {})],
    ('v1 (прежняя)', dict(size=13, bold=True, color=PURPLE)),
    '⅓·LightGBM + ⅓·CatBoost + ⅓·PI-MLP (физически-информированная нейросеть). PI-MLP убран: вне фолда не дал прироста.',
], size=10.5, spacing=1.05, space_after=6)
caption_bar(sl, 1.0, 14.1, 22.4, 3.5, [
    'Оба ансамбля работают поверх одного физического слоя. Итоговая оценка — взвешенное среднее логитов членов, сглаженное '
    'по 5 кадрам трека; STOP дальше 30 м — только при оценке ≥ порога и без вето оболочки тоннеля.'], size=11)

# ============================================================== ensemble weights
sl = base_slide('ВЕСА АНСАМБЛЯ')
figure_card(sl, os.path.join(DFIG, 'weight_sweep.png'), 1.0, 3.3, 31.9, 9.9)
cols = [
    ('Метод', 'Групповая проверка вне фолда: 8 групп (5 тоннелей + 3 блока новой линии), w₁ = 0…1 с шагом 0.1, '
              'w₂ = 1 − w₁. Порог каждого фолда — только на обучающих группах.'),
    ('Результат', 'Во всём диапазоне: полнота 67.3–67.6 %, ложные STOP 0.30–0.33 % — различия в пределах шума. '
                  'Модели согласованы, вес не влияет на качество.'),
    ('Решение', 'w₁ = w₂ = ½ выбраны заранее, без подгонки: минимум дисперсии и нет риска переобучить веса. '
                'Проверка подтвердила: оптимум плоский.'),
]
for i, (t, b_) in enumerate(cols):
    x = 1.0 + i * 10.8
    card(sl, x, 13.6, 10.3, 4.0, radius=0.07)
    text(sl, x + 0.4, 13.75, 9.5, 3.8, [(t, dict(bold=True, color=PURPLE, size=12.5)), b_], size=10.3, spacing=1.03,
         space_after=3)

# ============================================================== decision logic
sl = base_slide('ЛОГИКА РЕШЕНИЯ')
figure_card(sl, os.path.join(DFIG, 'decision_logic.png'), 1.0, 3.3, 26.0, 14.3)
card(sl, 27.5, 3.3, 5.4, 14.3, radius=0.06)
text(sl, 27.8, 3.7, 4.9, 13.6, [
    ('Выход', dict(size=13, bold=True, color=PURPLE)),
    'каждый кадр:', 'уровень 0 / 1 / 2', 'дистанция', 'время до столкновения', 'путь проверен до X м', 'все объекты',
    ('ROS 2 топики', dict(size=11, bold=True, color=PURPLE)), '/tunnel_guard/status', '/alarm', '/detections',
    '/markers'], size=9.5, spacing=1.0, space_after=3)

# ============================================================== synthetic objects: why and where
sl = base_slide('СИНТЕТИКА: ЗАЧЕМ')
numbered_cards(sl, [
    ('Измерить дальность', 'В данных нет реальных препятствий дальше ~60 м, а в датасете 2 — нет совсем. Человек или '
                           'коробка вставляются лучевым моделированием на 80–200 м — только так можно измерить, на каком '
                           'расстоянии система их находит.'),
    ('Позитивы для ML', 'Реальных препятствий мало, поэтому позитивные примеры для LightGBM/CatBoost — синтетические '
                        'объекты с точно известным положением. Негативы — реальные кадры без препятствий.'),
    ('Как', 'Для каждого реального луча Pandar128 считается пересечение с объектом: окклюзия, угловой шаг, дальность '
            'при данном отражении. Всё вокруг объекта — реальные данные записи.'),
    ('Только для оценки', 'Код — tools/ и core/synth.py. ROS-узел ничего не добавляет: данные заказчика и скрытый '
                          'тест обрабатываются как есть. В видео такие кадры помечены «СИНТЕТИКА».'),
])

# ============================================================== 18. demo: multiple frames
sl = base_slide('ДЕМОНСТРАЦИЯ')
frames = [
    ('demo_first_stop.png', f'1 · Датасет 1: реальный человек в габарите — STOP на {DEMO["holdout_first_stop"][1]:.1f} м'),
    ('demo_far.png', f'2 · Датасет 1, синтетика: человек — первый STOP на {DEMO["synthetic_first_stop"][1]:.0f} м'),
    ('demo_ds2_drive.png', '3 · Датасет 2: новая линия, 53 км/ч — путь проверен до 210 м'),
    ('demo_ds2_person.png', f'4 · Датасет 2, синтетика: человек на пути — STOP {DEMO["ds2_person_first_stop"][1]:.0f} м'),
]
fw, fh = 12.05, 6.78
for i, (fn, cap) in enumerate(frames):
    x = 1.0 + (i % 2) * (fw + 0.35)
    y = 3.25 + (i // 2) * (fh + 0.95)
    card(sl, x - 0.08, y - 0.08, fw + 0.16, fh + 0.16, fill=WHITE, line=None, radius=0.02)
    sl.shapes.add_picture(os.path.join(FIG, fn), Cm(x), Cm(y), Cm(fw), Cm(fh))
    on_purple(sl, x, y + fh + 0.1, fw, 0.8, [cap], size=10, bold=True)
px = 25.9
card(sl, px, 3.25, 7.0, 14.4, radius=0.05)
chain = [('Тоннель', 'реальный bag, 10 Гц'), ('Облако точек', 'до 921 600 лучей'), ('Алгоритм', 'путь → габарит → кластеры → гибрид'),
         ('Препятствие', 'рамка, зона, уверенность'), ('Расстояние', f'{R["hold_dist"]:.1f} м · время до столкновения')]
text(sl, px + 0.45, 3.55, 6.2, 1.2, [('Цепочка', dict(size=15, bold=True, color=PURPLE))])
for i, (t, b) in enumerate(chain):
    yy = 5.0 + i * 2.2
    text(sl, px + 0.45, yy, 1.2, 1.2, [f'{i + 1}'], size=20, bold=True, color=PINK)
    text(sl, px + 1.55, yy + 0.05, 5.2, 2.0, [(t, dict(bold=True, size=12.5)), (b, dict(size=10, color=MUTED))], spacing=1.0, space_after=1)
text(sl, px + 0.45, 16.1, 6.2, 1.4, [('Видео: docs/demo_tunnelguard.mp4', dict(size=10, bold=True, color=PURPLE))])

# ============================================================== 19. held-out real recording
sl = base_slide('ОТЛОЖЕННЫЙ ТЕСТ')
figure_card(sl, os.path.join(DFIG, 'holdout.png'), 1.0, 3.3, 21.6, 9.3)
caption_bar(sl, 1.0, 13.0, 21.6, 4.6, [
    [('Запись doubleT_obstacle ', dict(bold=True, color=PURPLE)), ('не использовалась ни при разработке, ни при настройке. '
     'Эталон положения людей — независимое вычитание фона стоящего поезда, а не детектор.', {})],
    [('Человек B ', dict(bold=True, color=PURPLE)), ('идёт рядом с поездом вне габарита — ни одного STOP (верно). '
     'CAUTION после ухода человека A — объекты у границы габарита, торможения не вызывают.', {})],
], size=11.5)
tiles = [(f'{R["hold_detected"]}/{R["hold_inside"]}', 'кадров с человеком в габарите — обнаружен'),
         (f'{R["hold_stop"]}', 'из них подтверждённый STOP'),
         (f'{R["hold_dist"]:.1f} м', 'дальность обнаружения человека'),
         (f'{R["hold_fp"]}', 'ложных STOP в записи'),
         (f'{R["hold_ms"]:.0f} мс', 'на кадр 921 600 лучей')]
for i, (v, l) in enumerate(tiles):
    kpi(sl, 23.1, 3.3 + i * 2.9, 9.8, 2.6, v, l, vsize=22, lsize=10.5)

# ============================================================== 20. detection range
sl = base_slide('ДАЛЬНОСТЬ')
figure_card(sl, os.path.join(DFIG, 'range.png'), 1.0, 3.3, 21.6, 10.2)
caption_bar(sl, 1.0, 13.9, 21.6, 3.7, [
    'Объекты вставлены лучевым моделированием в реальные лучи Pandar128 (окклюзия, угловой шаг, дальность при 10 % '
    'отражения — 200 м) и сближаются со скоростью 10 м/с; 3 старта × 5 записей. Полнота — по физически видимым случаям '
    '(≥ 3 отражений; за хордой кривой объект закрыт стеной).'], size=11)
tiles = [(f'{recall("person", 80.0):.0f} %', 'человек на 80 м'),
         (f'{recall("person", 120.0):.0f} %', 'человек на 120 м'),
         (f'{recall("person", 160.0):.0f} %', 'человек на 160 м (≈ 9 отражений)'),
         (f'{recall("box_50cm", 80.0):.0f} % · {recall("box_50cm", 120.0):.0f} %', 'коробка 0.5 м на 80 · 120 м'),
         (f'{DEMO["synthetic_first_stop"][1]:.0f} м', 'первый STOP синтетического человека в демо-ролике')]
for i, (v, l) in enumerate(tiles):
    kpi(sl, 23.1, 3.3 + i * 2.9, 9.8, 2.6, v, l, vsize=22, lsize=10.5)

# ============================================================== physical range limit
sl = base_slide('ПРЕДЕЛ ДАЛЬНОСТИ')
figure_card(sl, os.path.join(DFIG, 'range_budget.png'), 1.0, 3.3, 31.9, 10.4)
cols = [
    ('Коридор не виден дальше', 'В записях 99.9 % отражений ближе 110–140 м (медиана), путь измерен до 116–168 м; '
                                'ни в одном проверенном кадре нет отражений дальше 250 м: кривые, платформы, колонны.'),
    ('Физика сенсора', 'Паспорт Pandar128E3X: измеряемая дальность 200 м (в записях максимум 209 м); 200 м при 10 % отражения — только каналы 34–65. Дальше 210 м не измерить ничем: 300 м недостижимы для этого лидара.'),
    ('Проверили, а не поверили', 'Датасет v2 с объектами до 245 м, три политики дальнего STOP, leave-one-tunnel-out: '
                                 'из объектов на 150–245 м видны лишь 37, прироста полноты нет — прежняя модель оставлена.'),
]
for i, (t, b_) in enumerate(cols):
    x = 1.0 + i * 10.8
    card(sl, x, 14.1, 10.3, 3.5, radius=0.07)
    text(sl, x + 0.4, 14.25, 9.5, 3.3, [(t, dict(bold=True, color=PURPLE, size=12.5)), b_], size=10, spacing=1.03, space_after=3)

# ============================================================== 21. false alarms
sl = base_slide('ЛОЖНЫЕ ТРЕВОГИ')
figure_card(sl, os.path.join(DFIG, 'false_alarms.png'), 1.0, 3.3, 21.6, 10.8)
caption_bar(sl, 1.0, 14.5, 21.6, 3.1, [
    'Каждый кадр всех 5 записей без препятствий: кривые, станции, стрелка, гермозатворы. Оставшиеся 3 кадра — на записях '
    'со станциями и стрелкой; CAUTION у границы габарита информирует и не вызывает торможения.'], size=11)
tiles = [(f'{fp_loro:.2f} %', f'ложных STOP на невиденных тоннелях (правила: {fp_rules_loro:.2f} %)'),
         (f'{R["tot_stop"]} / {R["tot_frames"]}', 'кадров STOP на всех пустых записях (на обучающих)'),
         ('×4', 'меньше ложных остановок при росте полноты 62.8 → 64.5 %'),
         (f'{R["clear_med"]:.0f} м', 'медиана проверенной свободной дальности'),
         ('0', 'ложных STOP в отложенной записи с людьми')]
for i, (v, l) in enumerate(tiles):
    kpi(sl, 23.1, 3.3 + i * 2.9, 9.8, 2.6, v, l, vsize=22, lsize=10.5)

# ============================================================== new line: self-labelling
sl = base_slide('НОВАЯ ЛИНИЯ')
figure_card(sl, os.path.join(DFIG, 'ds2_traversal.png'), 1.0, 3.3, 21.6, 10.6)
caption_bar(sl, 1.0, 14.3, 21.6, 3.3, [
    '20 минут новой линии без разметки. Одометрия по «штрихкоду» стен (кронштейны, светильники, стыки колец) даёт '
    'координату каждой тревоги в тоннеле. Если поезд позже проехал эту точку — твёрдого объекта там не было: '
    'ложная тревога доказана физикой, независимо от проверяемого детектора.'], size=11)
tiles = [('13.0 км', 'одометрия по стенам: 0.000 м/кадр на стоящем поезде'),
         ('91 / 91', 'тревога старой модели проехана поездом — все ложные (≈ 7 на км)'),
         ('0', 'ручных меток: негативы доказаны, позитивы — лучевое моделирование'),
         ('2.8 км', 'последние 5 минут запечатаны: только финальный тест')]
for i, (v, l) in enumerate(tiles):
    kpi(sl, 23.1, 3.3 + i * 3.6, 9.8, 3.3, v, l, vsize=22, lsize=10.5)

# ============================================================== ego-motion test and sealed test
sl = base_slide('ЭГО-ДВИЖЕНИЕ')
figure_card(sl, os.path.join(DFIG, 'ds2_sealed.png'), 1.0, 3.3, 21.6, 10.6)
caption_bar(sl, 1.0, 14.3, 21.6, 3.3, [
    'Препятствие на пути приближается ровно со скоростью поезда: d(дистанция)/d(пробег) = −1. Артефакт, «едущий» с '
    'поездом (ошибка плоскости рельсов, рельс «на 0.3 м выше»), держит дистанцию ~25 м при 12 м/с: наклон ≈ 0 → '
    'только CAUTION. Стоящий поезд не отменяет ничего; обрыв одометрии сбрасывает доказательства.'], size=11)
tiles = [('×6', 'меньше ложных STOP-событий на запечатанном участке: 29 → 4.7'),
         ('1.8 / км', 'ложных STOP-событий финальной модели (было 10.3 / км), среднее по 3 зёрнам'),
         ('0', 'изменённых решений на исходных данных: полнота не потеряна'),
         ('56–70 мс', 'на кадр на одном ядре (лидар 10 Гц = 100 мс)')]
for i, (v, l) in enumerate(tiles):
    kpi(sl, 23.1, 3.3 + i * 3.6, 9.8, 3.3, v, l, vsize=22, lsize=10.5)

# ============================================================== dataset 3: organisers' synthetic obstacles
sl = base_slide('ДАТАСЕТ 3')
figure_card(sl, os.path.join(DFIG, 'ds3_scorecard.png'), 1.0, 3.3, 22.6, 11.0)
caption_bar(sl, 1.0, 14.7, 22.6, 2.9, [
    'Бэг организаторов: 10 синтетических объектов через ~100 м; читаются 29 % архива (438 кадров, 1.8 км) — в них все 10. '
    'Генератор ставит объекты на плоскость в системе лидара, а путь идёт под уклон: объекты «парят» над рельсами.'], size=10.5)
card(sl, 24.1, 3.3, 8.8, 14.3, radius=0.05)
text(sl, 24.5, 3.7, 8.0, 13.6, [
    ('Что изменили', dict(size=14, bold=True, color=PURPLE)),
    [('Трекер: ', dict(bold=True)), ('новый трек стартует со скоростью поезда; допуск на кадр задержки объекта.', {})],
    [('Одометрия: ', dict(bold=True)), ('переживает пропуски кадров до 1 с.', {})],
    [('Оболочка: ', dict(bold=True)), ('«прикреплён к своду» — только при непрерывности; висящее — не вето.', {})],
    [('Роутер: ', dict(bold=True)), ('парящие и висящие объекты оценивает эксперт, обученный на всех формах опасности.', {})],
    ('Итог', dict(size=14, bold=True, color=PURPLE)),
    'STOP на 7 из 8 объектов в габарите (куб 2×2 м — с 98 м вместо 48 м); не найден стержень 5 см. 1 ложное событие на '
    'пустом тоннеле за 1.8 км. Объекты 5 и 7 «за габаритом» дают STOP: наш габарит — вагон 2.7 м, их ≈ ±1.15 м от оси лидара '
    '(один параметр конфигурации — уточняем у организаторов).',
], size=10.3, spacing=1.04, space_after=5)

# ============================================================== 22. speed and resources
sl = base_slide('СКОРОСТЬ И РЕСУРСЫ')
figure_card(sl, os.path.join(DFIG, 'speed.png'), 1.0, 3.3, 21.6, 10.8)
caption_bar(sl, 1.0, 14.5, 21.6, 3.1, [
    'Векторизованная стоимость DP (доказано равна циклу), numba-ядра для DP и Гаусса–Ньютона (есть numpy-запасной путь), '
    'OpenBLAS в образе, JIT-кэш собирается в Docker-образе (загрузка ядер ≈ 1.4 с вместо 10 с).'], size=11)
tiles = [(f'{R["ms_med"]:.0f} мс', f'медиана на кадр · {1000 / R["ms_med"]:.0f} кадров/с (p95 {R["ms_p95"]:.0f} мс)'),
         ('1 ядро', 'CPU ноутбука i5-8250U 15 Вт; GPU не нужен'),
         ('≈ 0.2 ГБ', 'ОЗУ узла ROS 2 (при 128 ГБ на стенде)'),
         ('×3.9', 'ускорение: 268 → 69 мс; 552 → 89 мс на 921 600 лучах'),
         ('ROS 2 Humble', 'проверено: colcon build, тесты, evaluate_bag → STOP 55.5 м')]
for i, (v, l) in enumerate(tiles):
    kpi(sl, 23.1, 3.3 + i * 2.9, 9.8, 2.6, v, l, vsize=22, lsize=10.5)

# ============================================================== 23. evolution
sl = base_slide('ЭВОЛЮЦИЯ')
figure_card(sl, os.path.join(DFIG, 'evolution_stages.png'), 1.0, 3.3, 31.9, 10.6)
steps = [('Этапы 0–2 · геометрия', 'рельсы и стены: 249/252 ложных кадров; глобальная геометрия и σ(x): 90 → 89 на 5 записях'),
         ('Этап 3 · физика', 'гравитация, форма, стенка за объектом, крепление к оболочке: 89 → 34 кадра'),
         ('Этапы 4–5 · v1', '⅓ LGBM + ⅓ CatBoost + ⅓ PI-MLP, вето оболочки, numba: 34 → 3 кадра, 288 → 70 мс'),
         ('Этапы 6–7 · финал', 'новая линия и датасет 3: ЭГО-тест и роутер экспертов — 10.3 → 1.8 ложных остановок на км')]
for i, (t, b) in enumerate(steps):
    x = 1.0 + i * 8.05
    card(sl, x, 14.3, 7.7, 3.3, radius=0.08)
    text(sl, x + 0.35, 14.45, 7.0, 3.1, [(t, dict(bold=True, color=PURPLE, size=11.5)), b], size=10, spacing=1.02, space_after=3)

# ============================================================== metrics
sl = base_slide('МЕТРИКИ')
tiles = [('100 %', 'точность (precision) STOP: 58 из 58'), ('96.7 %', 'полнота (recall): 58 из 60 кадров'),
         ('99.0 %', 'accuracy по 201 кадру, F1 0.98')]
for i, (v, l) in enumerate(tiles):
    kpi(sl, 1.0 + i * 10.8, 3.3, 10.3, 3.6, v, l, vsize=30, lsize=11.5)
on_purple(sl, 1.0, 7.1, 31.9, 0.9, [('Реальная отложенная запись: позитив — человек A внутри габарита, прогноз — STOP. '
                                    'Финальная модель ½·LGBM + ½·CatBoost + ЭГО-тест; 2 пропуска — первые кадры до подтверждения (3 из 5).', dict(size=11.5))])
card(sl, 1.0, 8.2, 31.9, 9.4, radius=0.04)
rows = [('Классификатор кандидатов, leave-one-tunnel-out', 'Precision', 'Recall', 'Accuracy', 'Bal. acc.', 'ROC-AUC'),
        ('физические правила', '91.0 %', '73.0 %', '94.4 %', '85.8 %', '—'),
        ('LightGBM (порог 0.5)', '95.3 %', '81.7 %', '96.3 %', '90.5 %', '0.907'),
        ('CatBoost (порог 0.5)', '94.6 %', '80.3 %', '96.0 %', '89.7 %', '0.913'),
        ('ансамбль v1 (порог 0.37)', '79.3 %', '83.8 %', '93.7 %', '89.8 %', '0.904')]
widths = [12.4, 3.7, 3.7, 3.7, 3.7, 3.7]
for r_i, row in enumerate(rows):
    x = 1.5
    for c_i, cell in enumerate(row):
        st = dict(bold=(r_i == 0 or r_i == 4), color=PURPLE if r_i == 0 else (PINK if r_i == 4 else INK), size=12 if r_i else 11.5)
        text(sl, x, 8.55 + r_i * 1.3, widths[c_i], 1.2, [(cell, st)], align=PP_ALIGN.LEFT if c_i == 0 else PP_ALIGN.CENTER,
             anchor=MSO_ANCHOR.MIDDLE)
        x += widths[c_i]
text(sl, 1.5, 15.2, 30.9, 2.3, [
    'Одиночные кандидаты до сглаживания и подтверждения. Accuracy завышена: 84 % кандидатов — негативы. Порог 0.37 '
    'выбран ради полноты: сглаживание по треку и M из N убирают одиночные ложные кандидаты — на уровне решений '
    'ансамбль даёт 0.31 % ложных STOP-кадров при полноте 64.5 %.'], size=11, color=MUTED)

# ============================================================== ablation
if os.path.exists(os.path.join(DFIG, 'ablation_heatmap.png')):
    sl = base_slide('АБЛЯЦИЯ')
    figure_card(sl, os.path.join(DFIG, 'ablation_heatmap.png'), 1.0, 3.3, 21.4, 14.3)
    card(sl, 22.9, 3.3, 10.0, 14.3, radius=0.05)
    text(sl, 23.4, 3.7, 9.0, 13.6, [('Главные выводы', dict(size=15, bold=True, color=PURPLE))] + list(ABLATION_NOTES),
         size=11, spacing=1.08, space_after=7)

# ============================================================== leakage
sl = base_slide('БЕЗ УТЕЧЕК')
figure_card(sl, os.path.join(DFIG, 'leakage_cv.png'), 1.0, 3.3, 21.6, 14.3)
card(sl, 23.1, 3.3, 9.8, 14.3, radius=0.05)
text(sl, 23.6, 3.7, 8.8, 13.6, [
    ('Как исключена утечка', dict(size=15, bold=True, color=PURPLE)),
    [('Деление по записям и блокам. ', dict(bold=True)), ('Соседние кадры почти одинаковы — случайное деление по '
                                                          'кадрам было бы утечкой.', {})],
    [('Порог — только на обучении. ', dict(bold=True)), ('В каждом фолде порог выбирается внутри обучающих групп '
                                                         'под бюджет ложных STOP.', {})],
    [('Две записи не трогаем. ', dict(bold=True)), ('Люди (датасет 1) и последние 5 минут новой линии (B3) — '
                                                    'только финальный тест.', {})],
    [('Метки не от детектора. ', dict(bold=True)), ('Негативы доказаны проездом поезда, позитивы — лучевое '
                                                     'моделирование с точным положением.', {})],
    [('Честно: ', dict(bold=True, color=PINK)), ('тест эго-движения придуман после анализа ошибок B3; параметров, '
                                                 'подогнанных к B3, у него нет, на датасете 1 он не изменил ни одного '
                                                 'решения.', {})],
], size=10.5, spacing=1.05, space_after=6)

# ============================================================== 24. what did not work
sl = base_slide('ЧТО НЕ СРАБОТАЛО')
numbered_cards(sl, [
    ('Рельсы и стены', 'Рельсы видны лишь до 30–45 м, а стены на кривых дают ложную геометрию: 161 кадр тревоги в одной записи. '
                       'Решение: шаблон сечения + глобальный DP + плотный Гаусс–Ньютон.'),
    ('«Коридор должен быть пуст»', 'Штраф за точки в коридоре убрал колонны, но позволил «обойти» человека на 160 м. '
                                   'Решение: точки внутри габарита не влияют на геометрию.'),
    ('Слияние экстраполяций', 'Повторная экстраполяция выглядела как измерение и занижала σ — 120 ложных кадров. '
                              'Решение: во времени сливается только измеренная информация.'),
    ('Экспорт XGBoost и DDS', 'XGBoost после экспорта расходился на 4·10⁻² логита — исключён. Best-effort DDS терял '
                              'облака 9 МБ при ros2 bag play — launch включает reliable-воспроизведение.'),
])

# ============================================================== 25. limits and trade-offs
sl = base_slide('ОГРАНИЧЕНИЯ')
numbered_cards(sl, [
    ('Дальность упирается в физику', 'Pandar128 возвращает ≈ 200 м при 10 % отражения; на 160 м человек даёт ≈ 9 точек '
                                     'при шаге лучей 0.35 м. На кривой путь виден лишь до ≈ √(8RW).'),
    ('Мелкие объекты', 'Коробка 0.3 м уверенно подтверждается до 40–60 м, дальше — максимум CAUTION. '
                       'Осознанный компромисс против ложных остановок.'),
    ('ML обучен на синтетике', 'Позитивные примеры — лучевое моделирование. Нужны реальные препятствия; до 30 м решают '
                               'правила, вето оболочки абсолютно. Одна реальная запись с людьми.'),
])

# ============================================================== 26. launch and reproducibility
sl = base_slide('ЗАПУСК')
card(sl, 1.0, 3.3, 19.6, 14.3, fill=RGBColor(0x1C, 0x10, 0x2E), line=None, radius=0.035)
code = [
    ('# 1. сборка (все зависимости ставятся автоматически)', dict(color=RGBColor(0x9A, 0x8F, 0xB8))),
    ('docker build -t tunnel_guard .', dict(color=WHITE)),
    ('', {}),
    ('# 2. детектор + RViz2 + проигрывание bag', dict(color=RGBColor(0x9A, 0x8F, 0xB8))),
    ('docker run --rm -it --net=host -e DISPLAY=$DISPLAY \\', dict(color=WHITE)),
    ('  -v /tmp/.X11-unix:/tmp/.X11-unix -v /bags:/data tunnel_guard \\', dict(color=WHITE)),
    ('  ros2 launch tunnel_guard tunnel_guard.launch.py \\', dict(color=WHITE)),
    ('  bag:=/data/doubleT_obstacle rviz:=true', dict(color=RGBColor(0xFF, 0x7A, 0xA8))),
    ('', {}),
    ('# 3. результат', dict(color=RGBColor(0x9A, 0x8F, 0xB8))),
    ('ros2 topic echo /tunnel_guard/status', dict(color=WHITE)),
    ('', {}),
    ('# оффлайн-оценка bag быстрее реального времени', dict(color=RGBColor(0x9A, 0x8F, 0xB8))),
    ('ros2 run tunnel_guard evaluate_bag --bag /data/<bag>', dict(color=WHITE)),
]
text(sl, 1.6, 3.8, 18.6, 13.4, [(t, dict(st, font=MONO)) for t, st in code], size=11.5, spacing=1.0, space_after=2)
card(sl, 21.1, 3.3, 11.8, 14.3, radius=0.05)
text(sl, 21.6, 3.75, 10.9, 13.5, [
    ('Проверено', dict(size=15, bold=True, color=PURPLE)),
    'Ubuntu 22.04 + ROS 2 Humble: colcon build, 14 тестов pytest, ros2 bag play / evaluate_bag → STOP на 55.5 м',
    'Топик и ось лидара — автоматически; launch ждёт готовности узла и воспроизводит bag без потерь',
    ('Параметры', dict(size=15, bold=True, color=PURPLE)),
    'config/tunnel_guard.yaml: габарит, σ-множитель, M-из-N, дальности, модель скорера (none — только правила)',
    ('Документация', dict(size=15, bold=True, color=PURPLE)),
    'README · docs/EXPERIMENTS.md · tools/ воспроизводят каждое число в этой презентации',
], size=11.5, spacing=1.08, space_after=7)

# ============================================================== case requirements -> implementation
sl = base_slide('ТРЕБОВАНИЯ КЕЙСА')
reqs = [('Обработка облаков точек 3D-лидара', 'ROS 2 узел: любой PointCloud2, автоопределение топика и оси'),
        ('Мониторинг в реальном времени', '56–70 мс на кадр на 1 ядре CPU при лидаре 10 Гц'),
        ('Объекты в габарите поезда', 'габарит вдоль восстановленного пути с запасом 2σ'),
        ('Максимальная дальность', 'путь проверяется до 200 м (предел Pandar128); человек: 100 % на 80 м, 78 % на 120 м'),
        ('Мало ложных при высокой чувствительности', '0 ложных STOP на датасете 1, 1.8 на км новой линии; 58/60 кадров с человеком'),
        ('Выход для систем поезда', 'ObstacleStatus, Detection3DArray, alarm, дистанция, TTC, маркеры RViz'),
        ('ROS 2 узлы загрузки и обработки', 'detector_node, evaluate_bag, launch + ros2 bag play, Docker'),
        ('Фильтрация, сегментация, 3D-анализ', 'FOV и прорежение, геометрия пути, кластеризация вдоль пути, тесты формы'),
        ('Модуль обнаружения объектов', 'физика + роутер двух ансамблей + ЭГО-тест, трекер 3 из 5'),
        ('Интерфейсы и сообщения', 'пакет tunnel_guard_msgs: Obstacle.msg, ObstacleStatus.msg; стандартные vision_msgs')]
card(sl, 1.0, 3.3, 31.9, 14.3, radius=0.03)
text(sl, 1.6, 3.55, 12.0, 0.9, [('Требование кейса', dict(bold=True, color=PURPLE, size=12))])
text(sl, 14.2, 3.55, 17.0, 0.9, [('Как выполнено в TunnelGuard', dict(bold=True, color=PURPLE, size=12))])
for i, (rq, how) in enumerate(reqs):
    y = 4.55 + i * 1.28
    text(sl, 1.6, y, 12.3, 1.2, [[('●  ', dict(bold=True, color=PINK)), (rq, dict(bold=True))]], size=10.8,
         anchor=MSO_ANCHOR.MIDDLE)
    text(sl, 14.2, y, 18.3, 1.2, [how], size=10.8, anchor=MSO_ANCHOR.MIDDLE)

# ============================================================== 27. results summary
sl = base_slide('ИТОГ')
tiles = [('1.8 / км', 'ложных STOP на запечатанном участке новой линии (было 10.3)'),
         ('58/60', 'кадров с реальным человеком в габарите — STOP, 0 ложных'),
         (f'{R["hold_dist"]:.1f} м', 'реальный человек подтверждён STOP'),
         ('100 % · 78 %', 'полнота: человек на 80 м · 120 м'),
         (f'{R["ms_med"]:.0f} мс', f'на кадр, {1000 / R["ms_med"]:.0f} кадров/с на одном ядре'),
         (f'{DEMO["synthetic_first_stop"][1]:.0f} м', 'первый STOP синтетического человека в видео')]
for i, (v, l) in enumerate(tiles):
    x = 1.0 + (i % 3) * 10.8
    y = 3.4 + (i // 3) * 5.0
    kpi(sl, x, y, 10.3, 4.6, v, l, vsize=34, lsize=12.5)
card(sl, 1.0, 13.6, 31.9, 4.0, radius=0.1)
text(sl, 1.6, 13.6, 30.7, 4.0, [
    (f'«Вот поезд едет по тоннелю. Вот лидар. А вот препятствие, которое алгоритм увидел за {R["hold_dist"]:.1f} м» — ', dict(bold=True, color=PURPLE)),
    'и сказал, насколько далеко путь проверен свободным. Решение объяснимо, воспроизводимо (Docker + ROS 2) и обобщается: '
    'геометрия и габарит восстанавливаются из данных нового тоннеля, а не запоминаются.',
], size=13, anchor=MSO_ANCHOR.MIDDLE, spacing=1.1)

# ============================================================== 28. roadmap
sl = base_slide('РАЗВИТИЕ')
numbered_cards(sl, [
    ('Скорость', 'Порт геометрического ядра на C++/CUDA (на стенде кейса — RTX 4070 Ti SUPER): цель ≤ 20 мс на кадр и 20 Гц лидара.'),
    ('Слияние датчиков', 'Камера и радар подтверждают дальние объекты; лидар даёт геометрию пути для их проекции.'),
    ('Карта тоннеля', 'Накопление геометрии по поездкам — точный габарит, дальность до 300 м и мониторинг изменений инфраструктуры.'),
    ('Реальные данные', 'Запись реальных препятствий для калибровки ML; формальные требования к σ и тормозному пути; сертификация.'),
])

# ============================================================== remove guide and example slides (1-6, 12-37)
n_orig = 37
sld = prs.slides._sldIdLst
for i, sid in enumerate(list(sld)[:n_orig]):
    if not (7 <= i + 1 <= 11):
        prs.part.drop_rel(sid.rId)
        sld.remove(sid)
out = os.path.join(OUT_DIR, 'TunnelGuard_LCT2026.pptx')
prs.save(out)
print('saved', out, 'slides', len(prs.slides))
