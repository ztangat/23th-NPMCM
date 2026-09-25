const fs = require('fs');
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun, HeadingLevel, AlignmentType,
  WidthType, ShadingType, BorderStyle, LevelFormat, TableOfContents, Footer, PageNumber, PageBreak } = require('docx');
const [, , IN, OUT] = process.argv;
const blocks = JSON.parse(fs.readFileSync(IN, 'utf8'));
const FONT = { ascii: 'Times New Roman', eastAsia: 'SimSun', hAnsi: 'Times New Roman' };
const TW = 9360; // A4 内容宽(页边距 ~2.2cm)
const border = { style: BorderStyle.SINGLE, size: 4, color: '999999' };
const borders = { top: border, bottom: border, left: border, right: border };
const children = [];
const para = (text, opt = {}) => new Paragraph({ spacing: { after: 120, line: 360 }, indent: opt.noIndent ? undefined : { firstLine: 480 }, alignment: opt.align,
  children: [new TextRun({ text, font: FONT, size: opt.size || 22, italics: opt.italics, bold: opt.bold, color: opt.color })] });

children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 2400, after: 400 }, children: [new TextRun({ text: '2026年中国研究生数学建模竞赛 F题', font: FONT, size: 28 })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 400 }, children: [new TextRun({ text: '问题二求解报告', font: FONT, size: 44, bold: true })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 1200 }, children: [new TextRun({ text: '跨维度数据融合与广义标度律', font: FONT, size: 30 })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: '数据：附件 B1–B12 与问题一输出（A1 补全版）｜代码与过程数据见随附压缩包', font: FONT, size: 21, color: '555555' })] }));
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: '目录', font: FONT })] }));
children.push(new TableOfContents('目录', { hyperlink: true, headingStyleRange: '1-2' }));
children.push(new Paragraph({ children: [new PageBreak()] }));

let first = true;
for (const b of blocks) {
  if (b.t === 'h1') {
    if (!first) children.push(new Paragraph({ children: [new PageBreak()] }));
    first = false;
    children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: b.x, font: FONT })] }));
  } else if (b.t === 'h2') children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: b.x, font: FONT })] }));
  else if (b.t === 'h3') children.push(new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun({ text: b.x, font: FONT })] }));
  else if (b.t === 'p') children.push(para(b.x));
  else if (b.t === 'eq') children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 120 },
    children: [new TextRun({ text: b.x, font: { ascii: 'Cambria Math', eastAsia: 'SimSun', hAnsi: 'Cambria Math' }, size: 21, italics: true })] }));
  else if (b.t === 'note') children.push(new Paragraph({ spacing: { before: 60, after: 160, line: 340 }, shading: { type: ShadingType.CLEAR, fill: 'FFF4E0' },
    border: { left: { style: BorderStyle.SINGLE, size: 18, color: 'E69500', space: 6 } },
    children: [new TextRun({ text: b.x, font: FONT, size: 20 })] }));
  else if (b.t === 'bullets') for (const it of b.x) children.push(new Paragraph({ numbering: { reference: 'bul', level: 0 }, spacing: { after: 80, line: 340 },
    children: [new TextRun({ text: it, font: FONT, size: 21 })] }));
  else if (b.t === 'img') {
    const buf = fs.readFileSync(b.f);
    const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20); const W = 600; const H = Math.round(W * h / w);
    children.push(new Paragraph({ alignment: AlignmentType.CENTER, children: [new ImageRun({ type: 'png', data: buf, transformation: { width: W, height: H } })] }));
    children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: [new TextRun({ text: b.cap, font: FONT, size: 19, bold: true })] }));
  } else if (b.t === 'table') {
    const n = b.head.length; let ws = b.w;
    if (!ws || ws.length !== n) {
      const lens = b.head.map((h, j) => Math.min(40, Math.max(String(h).length, ...b.rows.map(r => String(r[j]).length))) + 3);
      const s = lens.reduce((a, c) => a + c, 0); ws = lens.map(l => Math.floor(TW * l / s));
    }
    const sum = ws.reduce((a, c) => a + c, 0); ws = ws.map(x => Math.floor(x * TW / sum)); ws[n - 1] += TW - ws.reduce((a, c) => a + c, 0);
    const sz = Math.round((b.fs || 8) * 2);
    const cell = (t, j, head) => new TableCell({ borders, width: { size: ws[j], type: WidthType.DXA }, margins: { top: 30, bottom: 30, left: 60, right: 60 },
      shading: head ? { type: ShadingType.CLEAR, fill: 'DCE6F1' } : undefined,
      children: [new Paragraph({ children: [new TextRun({ text: String(t), font: FONT, size: sz, bold: head })] })] });
    children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true, children: [new TextRun({ text: b.cap, font: FONT, size: 19, bold: true })] }));
    children.push(new Table({ width: { size: TW, type: WidthType.DXA }, columnWidths: ws,
      rows: [new TableRow({ tableHeader: true, children: b.head.map((h, j) => cell(h, j, true)) }), ...b.rows.map(r => new TableRow({ children: r.map((c, j) => cell(c, j, false)) }))] }));
    children.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
  }
}
const doc = new Document({
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { size: 32, bold: true, font: { ascii: 'Times New Roman', eastAsia: 'SimHei' } }, paragraph: { spacing: { before: 240, after: 200 }, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { size: 26, bold: true, font: { ascii: 'Times New Roman', eastAsia: 'SimHei' } }, paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 1 } },
      { id: 'Heading3', name: 'Heading 3', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { size: 23, bold: true }, paragraph: { spacing: { before: 160, after: 100 }, outlineLevel: 2 } }]
  },
  numbering: { config: [{ reference: 'bul', levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 600, hanging: 300 } } } }] }] },
  sections: [{ properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1300, bottom: 1300, left: 1270, right: 1270 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], size: 18 })] })] }) },
    children }]
});
Packer.toBuffer(doc).then(buf => { fs.writeFileSync(OUT, buf); console.log('ok', OUT); });
