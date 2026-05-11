#!/usr/bin/env bash

cd "$(dirname "$0")/.." || exit 1


## REC

# REC_example_1
# python demo/demo.py \
#   --modal image \
#   --model_path "weights/DeViL-7B" \
#   --media_path "assets/REC_example_1.jpg" \
#   --query "Locate the visual content described by the query <query>What's under the skier's feet?</query> in the image." \
#   --visualize_attention


# REC_example_2
# python demo/demo.py \
#   --modal image \
#   --model_path "weights/DeViL-7B" \
#   --media_path "assets/REC_example_2.jpg" \
#   --query "Locate the visual content described by the query <query>A little bear was being held.</query> in the image." \
#   --visualize_attention

# REC_example_3
# python demo/demo.py \
#   --modal image \
#   --model_path "weights/DeViL-7B" \
#   --media_path "assets/REC_example_3.jpg" \
#   --query "Locate the visual content described by the query <query>The sweetest food.</query> in the image." \
#   --visualize_attention

# ## STVG

# STVG_example_1
# python demo/demo.py \
#   --modal video \
#   --model_path "weights/DeViL-7B" \
#   --media_path "assets/STVG_example_1.mp4" \
#   --query "Locate the visual content described by the given textual query <query>The little boy was held in the arms of an adult.</query> in the video. Please output the start and end timestamps in seconds and the spatial location of the object." \
#   --visualize_attention


# STVG_example_1_alt_query
# python demo/demo.py \
#   --modal video \
#   --model_path "weights/DeViL-7B" \
#   --media_path "assets/STVG_example_1.mp4" \
#   --query "Locate the visual content described by the given textual query <query>What is the child holding when sitting on a doll?</query> in the video. Please output the start and end timestamps in seconds and the spatial location of the object." \
#   --visualize_attention

# STVG_example_2
# python demo/demo.py \
#   --modal video \
#   --model_path "weights/DeViL-7B" \
#   --media_path "assets/STVG_example_2.mp4" \
#   --query "Locate the visual content described by the given textual query <query>A child, held by an adult, kicked at the yellow, withered leaves on the ground.</query> in the video. Please output the start and end timestamps in seconds and the spatial location of the object." \
#   --visualize_attention

# STVG_example_3
# python demo/demo.py \
#   --modal video \
#   --model_path "weights/DeViL-7B" \
#   --media_path "assets/STVG_example_3.mp4" \
#   --query "Locate the visual content described by the given textual query <query>What is the child pushing in the room?</query> in the video. Please output the start and end timestamps in seconds and the spatial location of the object." \
#   --visualize_attention


# ## SVG

# SVG_example_1
# python demo/demo.py \
#   --modal video \
#   --model_path "weights/DeViL-7B" \
#   --media_path "assets/SVG_example_1.mp4" \
#   --query "Locate the visual content described by the given textual query <query>A little dog.</query> in the video." \
#   --visualize_attention


# SVG_example_1_alt_query
# python demo/demo.py \
#   --modal video \
#   --model_path "weights/DeViL-7B" \
#   --media_path "assets/SVG_example_1.mp4" \
#   --query "Locate the visual content described by the given textual query <query>What is the dog chewing on?</query> in the video." \
#   --visualize_attention

# ## TVG

# TVG_example_1
# python demo/demo.py \
#   --modal video \
#   --model_path "weights/DeViL-7B" \
#   --media_path "assets/TVG_example_1.mp4" \
#   --query "Locate the time of visual content described <query>A man opened the cabinet and looked inside.</query> Output the start and end timestamps in seconds." \

# TVG_example_2
# python demo/demo.py \
#   --modal video \
#   --model_path "weights/DeViL-7B" \
#   --media_path "assets/TVG_example_2.mp4" \
#   --query "Locate the time of visual content described <query>The woman is putting on a green dress.</query> Output the start and end timestamps in seconds." \


# ## GQA

# GQA_example_1
# python demo/demo.py \
#   --modal video \
#   --model_path "weights/DeViL-7B" \
#   --media_path "assets/GQA_example_1.mp4" \
#   --query "Answer only the questions I asked. What is the child in the yellow clothes holding?" \


# GQA_example_1_localization
# python demo/demo.py \
#   --modal video \
#   --model_path "weights/DeViL-7B" \
#   --media_path "assets/GQA_example_1.mp4" \
#   --query "Locate the visual content described by the given textual query <query>What is the child in the yellow clothes holding?</query> in the video. Please output the start and end timestamps in seconds and the spatial location of the object." \
#   --visualize_attention


# GQA_example_2
# python demo/demo.py \
#   --modal video \
#   --model_path "weights/DeViL-7B" \
#   --media_path "assets/GQA_example_2.mp4" \
#   --query "Answer only the questions I asked. What is the child in the yellow clothes holding?" \

# GQA_example_2_localization
# python demo/demo.py \
#   --modal video \
#   --model_path "weights/DeViL-7B" \
#   --media_path "assets/GQA_example_2.mp4" \
#   --query "Locate the visual content described by the given textual query <query>What is the child in the yellow clothes holding?</query> in the video. Please output the start and end timestamps in seconds and the spatial location of the object." \
#   --visualize_attention

# ## Video Understanding

# VideoUnderstanding_example_1
# python demo/demo.py \
#   --modal video \
#   --model_path "weights/DeViL-7B" \
#   --media_path "assets/VideoUnderstanding_example_1.mp4" \
#   --query "Please describe the video in detail." \


# VideoUnderstanding_example_1_qa
# python demo/demo.py \
#   --modal video \
#   --model_path "weights/DeViL-7B" \
#   --media_path "assets/VideoUnderstanding_example_1.mp4" \
#   --query "Answer only the questions I asked. Where is the scene in the video?" \