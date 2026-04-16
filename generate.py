import os
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='SDXL-Lightning CPU Inference')
    parser.add_argument('--prompt', type=str, required=True, help='生成 提示词')
    parser.add_argument('--steps', type=int, default=4, choices=[1, 2, 4, 8], help='推理步数')
    parser.add_argument('--num-images', type=int, default=1, help='生成 图片数量')
    parser.add_argument('--output-dir', type=str, default='outputs', help='输出目录')
    parser.add_argument('--height', type=int, default=512, help='图片高 度')
    parser.add_argument('--width', type=int, default=512, help='图片宽度')
    parser.add_argument('--model-path', type=str, default='./models', help='模型目录')
    parser.add_argument('--use-base-only', action='store_true', help='仅使用基础SDXL模型（不加载Lightning）')

    args = parser.parse_args()

    print(f"🚀 SDXL-Lightning CPU Inference")
    print(f"📝 Prompt: {args.prompt}")
    print(f"⚙️  Steps: {args.steps}")
    print(f"🖼️  Resolution: {args.height}x{args.width}")
    print(f"🔢 Count: {args.num_images}")
    print("-" * 50)

    # 延迟导入以加快启动
    import torch
    from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
    from safetensors.torch import load_file

    # 检查 CUDA 可用性
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 Using device: {device}")

    if device == "cpu":
        print("⚠️  CPU mode detected. Generation will be slower but works without GPU.")

    start_time = time.time()

    # 加载基础模型
    print("📦 Loading Stable Diffusion XL base model...")
    base_model = "stabilityai/stable-diffusion-xl-base-1.0"

    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
        variant="fp16",
        use_safetensors=True
    )

    # 加载 SDXL-Lightning 权重（如果存在且需要）
    if not args.use_base_only:
        lightning_model = f"sdxl_lightning_{args.steps}step.safetensors"
        model_file = Path(args.model_path) / lightning_model

        if model_file.exists():
            print(f"⚡ Loading SDXL-Lightning: {lightning_model}")
            state_dict = load_file(str(model_file))
            pipe.unet.load_state_dict(state_dict)
        else:
            print(f"⚠️  Lightning model not found: {model_file}")
            print("   Falling back to base SDXL model (slower)")

    # 配置调度器
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing"
    )

    # 移动到设备
    pipe = pipe.to(device)

    # 创建输出目录
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 生成图片
    generated_files = []

    for i in range(args.num_images):
        print(f"\n🎨 Generating image {i+1}/{args.num_images}...")
        img_start = time.time()

        image = pipe(
            prompt=args.prompt,
            num_inference_steps=args.steps,
            guidance_scale=0.0,  # Lightning 模型不需要 CFG
            height=args.height,
            width=args.width
        ).images[0]

        # 保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sdxl_{args.steps}step_{timestamp}_{i+1}.png"
        filepath = output_path / filename
        image.save(filepath)

        img_time = time.time() - img_start
        print(f"✅ Saved: {filepath} ({img_time:.2f}s)")
        generated_files.append(str(filepath))

    total_time = time.time() - start_time
    avg_time = total_time / args.num_images

    print("\n" + "=" * 50)
    print(f"🎉 Generation Complete!")
    print(f"⏱️  Total time: {total_time:.2f}s")
    print(f"🚀 Avg per image: {avg_time:.2f}s")
    print(f"📁 Output directory: {output_path.absolute()}")

    # GitHub Actions 输出
    if os.environ.get('GITHUB_ACTIONS') == 'true':
        with open(os.environ.get('GITHUB_OUTPUT', '/dev/null'), 'a') as f:
            f.write(f"files={','.join(generated_files)}\n")
            f.write(f"total_time={total_time:.2f}\n")
            f.write(f"avg_time={avg_time:.2f}\n")

        # 设置环境变量
        with open(os.environ.get('GITHUB_ENV', '/dev/null'), 'a') as f:
            f.write(f"SDXL_OUTPUT_DIR={output_path.absolute()}\n")

    return 0

if __name__ == "__main__":
    sys.exit(main())