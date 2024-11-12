import argparse
import tensorflow as tf
import os
from image_utils import load_image, save_image, show_image
from fgsm_attack import apply_fgsm_attack
from lpips_loss import LPIPSLossWithFGSM
from directory_manager import create_directory


def parse_args():
    parser = argparse.ArgumentParser(description="Generate adversarial images using FGSM.")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    parser.add_argument("output_path", type=str, help="Path to save the output image.")
    parser.add_argument('--epsilon', type=float, default=0.01, help="Epsilon value for FGSM attack")
    return parser.parse_args()

def main():
    args = parse_args()  # 명령줄 인자 받기
    image_path = args.image_path  # 이미지 경로
    output_path = args.output_path  # 출력 경로
    epsilon = args.epsilon

    # 디렉토리 생성 (필요한 경우)
    output_dir = os.path.dirname(output_path)
    create_directory(output_dir)
    
    # 이미지 불러오기
    image = load_image(image_path)
    image = tf.expand_dims(image, axis=0)  # 배치 차원 추가 (배치 크기 1)

    # FGSM 공격 적용
    lpips_fgsm_loss = LPIPSLossWithFGSM(trunk_network="alex", epsilon=0.01)
    adversarial_image, loss_value = apply_fgsm_attack(lpips_fgsm_loss, image, epsilon)
    
    # 결과 출력
    if adversarial_image.shape[0] == 1:  # 배치 차원만 검사
        adversarial_image_numpy = tf.squeeze(adversarial_image).numpy()  # 배치 차원 제거

        # 이미지 저장
        save_image(adversarial_image_numpy, output_path)

        # 이미지 시각화
        show_image(adversarial_image_numpy)
    else:
        print(f"Invalid shape for adversarial image: {adversarial_image.shape}")

    # 손실 값 출력
    print("Loss value:", loss_value.numpy())

if __name__ == "__main__":
    main()
