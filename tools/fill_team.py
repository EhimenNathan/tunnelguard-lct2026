"""Fill the team slides (1-4) of the built deck from team.json and member photos.

usage: python fill_team.py deck.pptx team_dir [out.pptx]
team_dir holds team.json and the photos it names; "head" = (x of the face centre, y of the top of the hair) in the
photo, used to crop head-and-shoulders portraits.  Slide 3 keeps as many member cards as there are members (centred),
the template's other cards are removed.
"""
import copy
import json
import os
import sys
import tempfile

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Emu, Pt

INK = RGBColor(0x1C, 0x1D, 0x22)
PURPLE = RGBColor(0x52, 0x09, 0x77)

A = '{http://schemas.openxmlformats.org/drawingml/2006/main}'


def set_runs(par, texts):
    """Put texts into the paragraph's runs, keeping each run's formatting; extra runs are removed."""
    runs = par.runs
    for r, t in zip(runs, texts):
        r.text = t
    for r in runs[len(texts):]:
        r._r.getparent().remove(r._r)
    for t in texts[len(runs):]:                      # more texts than runs: copy the last run's formatting
        new = copy.deepcopy(par.runs[-1]._r)
        par._p.append(new)
        par.runs[-1].text = t


def set_lines(par, lines):
    """One run per line separated by line breaks, all with the formatting of the paragraph's first run."""
    first = par.runs[0]._r
    for r in par.runs[1:]:
        r._r.getparent().remove(r._r)
    for el in par._p.findall(A + 'br'):
        par._p.remove(el)
    par.runs[0].text = lines[0]
    anchor = first
    for line in lines[1:]:
        br = copy.deepcopy(first)
        br.tag = A + 'br'
        for child in list(br):
            if child.tag != A + 'rPr':
                br.remove(child)
        run = copy.deepcopy(first)
        anchor.addnext(br)
        br.addnext(run)
        run.find(A + 't').text = line
        anchor = run


def crop(path, head, aspect, width, above):
    im = Image.open(path).convert('RGB')
    cx, top = head
    h = int(width / aspect)
    x0 = max(0, min(im.width - width, cx - width // 2))
    y0 = max(0, min(im.height - h, top - above))
    return im.crop((x0, y0, x0 + width, y0 + h))


def save_tmp(im, name):
    p = os.path.join(tempfile.gettempdir(), name)
    im.save(p, quality=92)
    return p


def shape(slide, sid):
    return next(s for s in slide.shapes if s.shape_id == sid)


def main(deck, team_dir, out=None):
    T = json.load(open(os.path.join(team_dir, 'team.json'), encoding='utf-8'))
    M = T['members']
    cap = next(m for m in M if m.get('captain'))
    prs = Presentation(deck)
    s1, s2, s3, s4 = (prs.slides[i] for i in range(4))

    # slide 1: title
    set_runs(shape(s1, 3).text_frame.paragraphs[0], [T['team']])

    # slide 2: team name, the "about the team" block and a group photo; the right column sits on a light background
    title = shape(s2, 18)
    set_runs(title.text_frame.paragraphs[0], [T['team']])
    title.text_frame.paragraphs[0].runs[0].font.color.rgb = PURPLE
    for sid in (5, 8):
        for par in shape(s2, sid).text_frame.paragraphs:
            for r in par.runs:
                r.font.color.rgb = INK
    n = len(M)
    word = 'человек' if n % 10 not in (2, 3, 4) or n % 100 in (12, 13, 14) else 'человека'
    ps = shape(s2, 14).text_frame.paragraphs
    set_runs(ps[0], ['Капитан: ', f"{' '.join(cap['name'])}, {cap['role'].split(',')[0]}"])
    set_runs(ps[1], ['Кол-во участников: ', f'{n} {word}'])
    set_runs(ps[3], [f"Пришли из разных областей и познакомились в {T['org']}, где создаём сложные "
                     "продукты компьютерного зрения и машинного обучения."])
    set_runs(ps[4], [f"Место работы участников: {T['org']}."])
    set_runs(ps[5], ['Город и регион: ', T['region']])
    ps[5].runs[1].font.bold = False
    ph = shape(s2, 2)
    box = (ph.left, ph.top, ph.width, ph.height)
    panels = [crop(os.path.join(team_dir, m['photo']), m['head'], box[2] / box[3] / n, 1000, 90) for m in M]
    pw = 650
    group = Image.new('RGB', (pw * n, int(pw * panels[0].height / panels[0].width)))
    for k, im in enumerate(panels):
        group.paste(im.resize((pw, group.height), Image.LANCZOS), (k * pw, 0))
    pic = ph.insert_picture(save_tmp(group, 'tg_group.jpg'))
    pic.left, pic.top, pic.width, pic.height = (Emu(v) for v in box)

    # slide 3: member cards (template: 5 cards; keep n, centred)
    cards = [(17, 2, 9, 15), (56, 3, 57, 58), (59, 4, 60, 61), (62, 5, 63, 64), (65, 6, 66, 67)]
    pitch = 2305318
    left0 = 346076 + (5 - n) * pitch // 2
    keep = cards[(5 - n) // 2:(5 - n) // 2 + n]
    for c in cards:
        if c not in keep:
            for sid in c:
                el = shape(s3, sid)._element
                el.getparent().remove(el)
    nb = lambda t: t.replace(' ', '\u00a0')     # phone numbers and handles never wrap
    for k, (m, (box_id, pic_id, info, name)) in enumerate(zip(M, keep)):
        x = left0 + k * pitch
        shape(s3, box_id).left = x
        img = crop(os.path.join(team_dir, m['photo']), m['head'], 1722566 / 1552901, 1150, 70)
        pp = shape(s3, pic_id).insert_picture(save_tmp(img, f'tg_card{k}.jpg'))
        pp.left, pp.top, pp.width, pp.height = Emu(x + 235073), Emu(1816054), Emu(1722566), Emu(1552901)
        nm = shape(s3, name)
        nm.left, nm.width, nm.height = Emu(x + 172284), Emu(1894354), Emu(760000)
        words = m['name'][0].split() + m['name'][1].split() if len(' '.join(m['name'])) > 18 else m['name']
        set_lines(nm.text_frame.paragraphs[0], words)
        inf = shape(s3, info)
        inf.left, inf.top, inf.width = Emu(x + 110000), Emu(4400000), Emu(1990000)
        lines = (['Капитан команды'] if m.get('captain') else []) + \
            [m['role'], m['tg'], nb(m['phone']), T['org']]
        pars = inf.text_frame.paragraphs
        while len(pars) < len(lines):
            pars[-1]._p.addnext(copy.deepcopy(pars[-1]._p))
            pars = inf.text_frame.paragraphs
        for par, t in zip(pars, lines):
            set_runs(par, [t])
            par.runs[0].font.size = Pt(11)
        if m.get('captain'):
            pars[0].runs[0].font.bold = True

    # slide 4: 01 = how the team met, 02 = why this task; body texts sit on a light background
    set_runs(shape(s4, 37).text_frame.paragraphs[0], [
        f"Мы пришли из разных областей и познакомились в {T['org']}, где вместе создаём сложные продукты "
        "компьютерного зрения и машинного обучения. Разный опыт помогает смотреть на задачу с нескольких сторон — "
        "от физики сенсора до промышленного ML."])
    set_runs(shape(s4, 43).text_frame.paragraphs[0], [
        "Метро перевозит миллионы людей в день, и беспилотный поезд должен видеть путь надёжнее человека: пропуск "
        "препятствия — угроза жизни, ложная тревога — остановка всей линии. Нам близки задачи, где компьютерное "
        "зрение работает в жёстких условиях — темнота тоннеля, кривые, стрелки, решение за 100 мс. Мы хотели "
        "показать, что объяснимая физика тоннеля вместе с ML даёт систему, которой можно доверить поезд с пассажирами."])
    for sid in (37, 40, 43):
        sh = shape(s4, sid)
        sh.width = Emu(10200000)
        for par in sh.text_frame.paragraphs:
            for r in par.runs:
                r.font.color.rgb = INK
                r.font.size = Pt(12)

    prs.save(out or deck)
    print('saved', out or deck)


if __name__ == '__main__':
    main(*sys.argv[1:])
