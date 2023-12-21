configdate=$(date '+%Y-%m-%d')
cd /path/to/clinical-trial-ie/
mkdir -p ./results_"$configdate"/gen/dm2-t5
mkdir -p ./results_"$configdate"/gen/dm2-led
mkdir -p ./results_"$configdate"/gen/gl-t5
mkdir -p ./results_"$configdate"/gen/gl-led

mkdir -p ./results_"$configdate"/extr/dm2-t5
mkdir -p ./results_"$configdate"/extr/dm2-led
mkdir -p ./results_"$configdate"/extr/dm2-longformer
mkdir -p ./results_"$configdate"/extr/gl-t5
mkdir -p ./results_"$configdate"/extr/gl-led
mkdir -p ./results_"$configdate"/extr/gl-longformer

mv ./*gen_model_dm2_flan-t5* ./config_gen_dm2_flan-t5* ./results_"$configdate"/gen/dm2-t5/
mv ./*gen_model_dm2_led* ./config_gen_dm2_led* ./results_"$configdate"/gen/dm2-led/
mv ./*gen_model_gl_flan-t5* ./config_gen_gl_flan-t5* ./results_"$configdate"/gen/gl-t5/
mv ./*gen_model_gl_led* ./config_gen_gl_led* ./results_"$configdate"/gen/gl-led/

mv ./config_extr_dm2_flan-t5* ./extr_*_dm2_flan-t5* ./results_"$configdate"/extr/dm2-t5
mv ./config_extr_dm2_led* ./extr_*_dm2_led* ./results_"$configdate"/extr/dm2-led
mv ./config_extr_dm2_longformer* ./extr_*_dm2_longformer* ./results_"$configdate"/extr/dm2-longformer
mv ./config_extr_gl_flan-t5* ./extr_*_gl_flan-t5* ./results_"$configdate"/extr/gl-t5
mv ./config_extr_gl_led* ./extr_*_gl_led* ./results_"$configdate"/extr/gl-led
mv ./config_extr_gl_longformer* ./extr_*_gl_longformer* ./results_"$configdate"/extr/gl-longformer
