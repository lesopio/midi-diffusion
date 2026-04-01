import gradio as gr

from scripts import ui_shared as core

THEME = core.THEME
CSS = core.CSS


def build_ui():
    with gr.Blocks(title="MIDI ComfyUI") as demo:
        gr.Markdown("# MIDI ComfyUI")
        gr.Markdown("节点式流程编排：从导入到清洗、训练、分析与生成。")

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Accordion("输入节点", open=True):
                    with gr.Row():
                        zip_file = gr.File(label="导入 MIDI Zip", file_types=[".zip"])
                        out_dir = gr.Textbox(label="解压目录", value="data/midi")
                    clear_dir = gr.Checkbox(label="清空目标目录", value=False)
                    import_btn = gr.Button("解压导入")
                    gr.Markdown("或使用已有目录作为数据源")
                    existing_dir = gr.Textbox(label="已有数据目录", value="data/midi")
                    use_btn = gr.Button("使用目录")
                    import_log = gr.Textbox(label="导入日志", lines=6)

                with gr.Accordion("清洗节点", open=False):
                    data_dir = gr.Textbox(label="原始数据目录", value="data/midi")
                    output_dir = gr.Textbox(label="输出目录", value="data/cleaned_dataset")
                    zip_path = gr.Textbox(label="输出 Zip 路径", value="data/cleaned_dataset.zip")
                    min_note_length = gr.Number(label="最小音符长度", value=0.05)
                    max_size_mb = gr.Number(label="最大文件大小(MB)", value=10)
                    keep_structure = gr.Checkbox(label="保留目录结构", value=False)
                    skip_zip = gr.Checkbox(label="跳过打包 zip", value=False)
                    run_pre_btn = gr.Button("开始清洗")
                    pre_log = gr.Textbox(label="清洗日志", lines=10)

                with gr.Accordion("训练节点", open=True):
                    with gr.Row():
                        data_dir_t = gr.Textbox(label="数据目录", value="data/midi")
                        use_zip = gr.Checkbox(label="使用清洗 zip", value=False)
                    data_zip = gr.Textbox(label="Zip 路径", value="data/cleaned_dataset.zip")
                    extract_dir = gr.Textbox(label="解压目录", value="data/packed_dataset")
                    save_dir = gr.Textbox(label="保存目录", value="checkpoints")

                    with gr.Row():
                        device = gr.Dropdown(choices=["4090", "4050"], value="4090", label="设备")
                        epochs = gr.Number(label="训练轮数", value=50)
                        batch_size = gr.Number(label="批大小(0=自动)", value=0)
                        steps_per_beat = gr.Number(label="每拍步数(0=自动)", value=0)
                    with gr.Row():
                        max_voices = gr.Number(label="最大声部(0=自动)", value=0)
                        aux_weight = gr.Number(label="Aux 权重", value=0.2)
                        save_every_steps = gr.Number(label="保存步数间隔", value=500)
                        resume = gr.Checkbox(label="断点恢复", value=False)
                    with gr.Row():
                        dashboard = gr.Checkbox(label="启用训练监控", value=True)
                        dynamic_length = gr.Checkbox(label="动态长度", value=True)
                        max_seq_len = gr.Number(label="最大序列长度(0=自动)", value=0)
                        bucket_size = gr.Number(label="分桶粒度", value=64)
                    with gr.Row():
                        val_split = gr.Number(label="验证集比例", value=0.05)
                        val_every_epochs = gr.Number(label="验证频率(epochs)", value=1)
                        eval_data_dirs = gr.Textbox(label="额外评估集(逗号分隔)", value="")
                        eval_max_samples = gr.Number(label="评估批次数上限(0=不限制)", value=0)

                    train_btn = gr.Button("开始训练")
                    train_progress = gr.HTML(
                        value=(
                            '<div style="height:10px;background:#E2E5E8;border-radius:0;overflow:hidden;">'
                            '<div style="width:0%;height:100%;background:#8EA6B5;"></div></div>'
                        )
                    )
                    train_status = gr.Textbox(label="训练状态", lines=1)
                    train_log = gr.Textbox(label="训练日志", lines=14)

                with gr.Accordion("分析节点", open=False):
                    gr.Markdown("统计长度/节奏/复调，并可对比两个数据集")
                    ana_data_dir = gr.Textbox(label="数据目录", value="data/midi")
                    ana_compare_dir = gr.Textbox(label="对比目录(可选)", value="")
                    ana_steps = gr.Number(label="Steps/Beat", value=4)
                    ana_max_len = gr.Number(label="最大序列长度(可选)", value=0)
                    ana_limit = gr.Number(label="采样上限(0=不限制)", value=0)
                    ana_out = gr.Textbox(label="输出报告路径", value="dataset_report.json")
                    ana_btn = gr.Button("开始分析")
                    ana_log = gr.Textbox(label="分析日志", lines=8)

                with gr.Accordion("生成节点", open=False):
                    gr.Markdown("从 checkpoint 生成 MIDI")
                    gen_ckpt = gr.Textbox(label="Checkpoint", value="checkpoints/checkpoint_latest.pt")
                    gen_out = gr.Textbox(label="输出目录", value="outputs")
                    gen_num = gr.Number(label="生成数量", value=1)
                    gen_seq = gr.Number(label="Seq Len", value=256)
                    gen_steps = gr.Number(label="采样步数", value=50)
                    gen_voices = gr.Number(label="最大声部", value=4)
                    gen_spb = gr.Number(label="Steps/Beat", value=4)
                    gen_tempo = gr.Number(label="Tempo", value=120.0)
                    gen_temp = gr.Number(label="Temperature", value=1.0)
                    gen_btn = gr.Button("开始生成")
                    gen_log = gr.Textbox(label="生成日志", lines=8)

            with gr.Column(scale=1):
                gr.Markdown("训练曲线")
                train_metrics = gr.HTML(
                    value=core._render_metrics_html(
                        {"loss": [], "diffusion": [], "lr": [], "throughput": [], "vram": []}
                    )
                )
                gr.Markdown("全局终端")
                terminal_all = gr.Textbox(label="总终端", lines=24, value=core._get_global_terminal())

        import_btn.click(
            fn=core.import_zip,
            inputs=[zip_file, out_dir, clear_dir],
            outputs=[import_log, out_dir, terminal_all],
        )
        use_btn.click(
            fn=core.use_existing_dir,
            inputs=[existing_dir],
            outputs=[import_log, out_dir, terminal_all],
        )
        run_pre_btn.click(
            fn=core.run_preprocess,
            inputs=[data_dir, output_dir, zip_path, min_note_length, max_size_mb, keep_structure, skip_zip],
            outputs=[pre_log, terminal_all],
        )
        train_btn.click(
            fn=core.run_train,
            inputs=[
                data_dir_t,
                data_zip,
                use_zip,
                extract_dir,
                save_dir,
                device,
                epochs,
                batch_size,
                steps_per_beat,
                max_voices,
                aux_weight,
                dashboard,
                resume,
                save_every_steps,
                dynamic_length,
                max_seq_len,
                bucket_size,
                val_split,
                val_every_epochs,
                eval_data_dirs,
                eval_max_samples,
            ],
            outputs=[train_log, train_progress, train_status, train_metrics, terminal_all],
        )
        ana_btn.click(
            fn=core.run_analysis,
            inputs=[ana_data_dir, ana_compare_dir, ana_steps, ana_max_len, ana_limit, ana_out],
            outputs=[ana_log, terminal_all],
        )
        gen_btn.click(
            fn=core.run_generate,
            inputs=[gen_ckpt, gen_out, gen_num, gen_seq, gen_steps, gen_voices, gen_spb, gen_tempo, gen_temp],
            outputs=[gen_log, terminal_all],
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.queue().launch(server_name="0.0.0.0", server_port=7860, theme=THEME, css=CSS)
