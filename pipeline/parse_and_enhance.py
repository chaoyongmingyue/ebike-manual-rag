"""pdf_md_enhancer - 使用本地 VLM (Qwen3-VL) 增强 MinerU 输出的 Markdown."""

import json
import os
import sys
import time
from pathlib import Path

from vlm_client import VLMClient
from pdf_cropper import PDFCropper

# Base paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / ".." / "data"


def build_vlm_prompt(item_type):
    prompts = {
        "image": (
            "请详细描述这张图片中的内容。如果包含文字，请完整提取所有文字。"
            "如果是产品照片或插图，请描述图中展示的物品、场景和关键信息。请用中文回答。"
        ),
        "table": (
            "这是一张表格图片，请完整提取表格中的所有文字内容，"
            "并按原格式输出为Markdown表格。请用中文回答。"
        ),
        "chart": (
            "这是一张图表或示意图，请描述其结构和展示的关键信息。"
            "如果包含文字请完整提取。请用中文回答。"
        ),
    }
    return prompts.get(item_type, "请详细描述这张图片的内容，包括所有文字信息。请用中文回答。")


def process_document(
    mineru_out_dir,
    pdf_path,
    output_md_path,
    vlm_model="qwen3-vl:4b",
    vlm_base_url="http://localhost:11434",
    skip_vlm=False,
):
    mineru_dir = Path(mineru_out_dir)
    auto_dir = mineru_dir / "auto"
    base_name = mineru_dir.name

    content_list_path = auto_dir / f"{base_name}_content_list_v2.json"
    images_dir = auto_dir / "images"
    crop_dir = auto_dir / "vlm_crops"

    if not content_list_path.exists():
        print(f"Error: {content_list_path} not found")
        sys.exit(1)

    print("Reading content list...")
    with open(content_list_path, "r", encoding="utf-8") as f:
        pages = json.load(f)

    vlm = None
    if not skip_vlm:
        vlm = VLMClient(model=vlm_model, base_url=vlm_base_url)
        print(f"VLM: {vlm_model} @ {vlm_base_url}")

    cropper = None
    if Path(pdf_path).exists():
        cropper = PDFCropper(pdf_path, dpi=150)
        crop_dir.mkdir(parents=True, exist_ok=True)
        print(f"PDF cropper ready: {pdf_path}")
    else:
        print(f"Warning: PDF not found at {pdf_path}")

    md_lines = []
    total_nontext = 0
    total_vlm_calls = 0
    t_start = time.time()

    for page_idx, page_items in enumerate(pages):
        doc_page_num = page_idx + 1
        md_lines.append(f"\n---\n<!-- Page {doc_page_num} -->\n")

        for item_idx, item in enumerate(page_items):
            item_type = item.get("type", "unknown")
            content = item.get("content", {})

            # ========== TEXT ELEMENTS ==========
            if item_type in ("paragraph", "title", "list", "index",
                             "page_header", "page_footer", "page_number"):
                text_parts = []
                for key, val in content.items():
                    if isinstance(val, list):
                        for part in val:
                            if isinstance(part, dict) and part.get("type") == "text":
                                text_parts.append(part["content"])
                            elif isinstance(part, str):
                                text_parts.append(part)
                    elif isinstance(val, str):
                        text_parts.append(val)

                text = "".join(text_parts).strip()
                if text:
                    if item_type == "title":
                        level = content.get("level", 2)
                        md_lines.append(f"{'#' * level} {text}\n")
                    elif item_type in ("page_header", "page_footer", "page_number"):
                        continue  # skip page numbers/headers
                    elif item_type == "list":
                        md_lines.append(f"- {text}\n")
                    else:
                        md_lines.append(f"{text}\n")

            # ========== NON-TEXT ELEMENTS ==========
            elif item_type in ("image", "table", "chart", "code"):
                total_nontext += 1
                img_abs = ""
                has_html = False
                need_crop = False

                img_source = content.get("image_source", {}) or {}
                img_rel = ""
                if isinstance(img_source, dict):
                    img_rel = img_source.get("path", "") or ""
                elif isinstance(img_source, str):
                    img_rel = img_source

                # Check extracted image
                if img_rel and img_rel.strip() and img_rel != "images/":
                    for cand in [images_dir / Path(img_rel).name, images_dir / img_rel]:
                        if Path(cand).exists():
                            img_abs = str(cand)
                            break

                # Check HTML table
                table_html = content.get("html", "") or ""
                has_html = bool(table_html.strip())

                # If no image found and no HTML, need to render page
                if not img_abs and not has_html:
                    need_crop = True

                # Get or generate description
                description = ""

                if has_html:
                    description = table_html.strip()
                elif need_crop and cropper:
                    # Render full page as image
                    crop_filename = f"page{doc_page_num}_full.jpg"
                    crop_path = str(crop_dir / crop_filename)
                    try:
                        # Render full page
                        pt_w, pt_h = cropper.get_page_size_pts(page_idx)
                        bitmap = cropper.pdf[page_idx].render(
                            scale=150/72,
                            rotation=0,
                        )
                        pil_img = bitmap.to_pil()
                        pil_img.save(crop_path, quality=90)
                        if Path(crop_path).exists():
                            img_abs = crop_path
                            print(f"  [fullpage] P{doc_page_num} {item_type} -> {crop_filename}")
                    except Exception as e:
                        print(f"  [fullpage fail] P{doc_page_num}: {e}")

                if not description and img_abs and vlm and not skip_vlm:
                    prompt = build_vlm_prompt(item_type)
                    fname = Path(img_abs).name
                    print(f"  [VLM] P{doc_page_num} {item_type} ({fname})...", end=" ", flush=True)
                    description = vlm.describe(img_abs, prompt)
                    total_vlm_calls += 1
                    print("OK")
                elif not description:
                    description = f"[{item_type} on page {doc_page_num}]"

                # Write to Markdown
                md_lines.append(f"\n<!-- {item_type} from page {doc_page_num} -->\n")

                if description and not description.startswith("[VLM Error"):
                    if has_html:
                        md_lines.append(f"{description}\n")
                    else:
                        md_lines.append(f"> **{item_type.capitalize()}**: {description}\n\n")

                if img_abs and item_type == "image":
                    rel_path = os.path.relpath(img_abs, str(auto_dir))
                    md_lines.append(f"![]({rel_path})\n")

            # ========== UNKNOWN ==========
            else:
                text_parts = []
                if isinstance(content, dict):
                    for key, val in content.items():
                        if isinstance(val, list):
                            for part in val:
                                if isinstance(part, dict) and part.get("type") == "text":
                                    text_parts.append(part["content"])
                                elif isinstance(part, str):
                                    text_parts.append(part)
                        elif isinstance(val, str):
                            text_parts.append(val)
                text = "".join(text_parts).strip()
                if text:
                    md_lines.append(f"{text}\n")

    if cropper:
        cropper.close()

    output_path = Path(output_md_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(md_lines))

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Done! Output: {output_path}")
    print(f"Pages: {len(pages)}, Non-text: {total_nontext}, VLM calls: {total_vlm_calls}")
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="使用VLM增强MinerU输出的Markdown")
    parser.add_argument("--mineru-out", default=str(DATA_DIR))
    parser.add_argument("--pdf", default=str(DATA_DIR / "电动自行车说明书_origin.pdf"))
    parser.add_argument("--output", default=str(DATA_DIR / "电动自行车说明书_enhanced.md"))
    parser.add_argument("--vlm-model", default="qwen3-vl:4b")
    parser.add_argument("--vlm-url", default="http://localhost:11434")
    parser.add_argument("--skip-vlm", action="store_true")
    args = parser.parse_args()
    process_document(args.mineru_out, args.pdf, args.output, args.vlm_model, args.vlm_url, args.skip_vlm)
