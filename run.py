import training
import prompts.ai_questions
import prompts.av_prompt
import prompts.paraphrase_prompt
import prompts.semantic_meaning_prompt

batch_size = 4
GRPO_size = 8



av, ar, activation_model, av_optimizer, ar_optimizer, ar_parameters = training.setup()

training.train(ai_prompts=prompts.ai_questions.ai_prompts, 
               paraphrase_prompt=prompts.paraphrase_prompt.para_prompt, 
               av_prompt=prompts.av_prompt.av_prompt_, 
               semantic_meaning_prompt=prompts.semantic_meaning_prompt.sm_prompt, 
               activation_model=activation_model, 
               av=av, 
               ar=ar, 
               av_optimizer=av_optimizer, 
               ar_optimizer=ar_optimizer, 
               total_steps=5000, 
               ar_parameters=ar_parameters, 
               batch_size=batch_size, 
               GRPO_size=GRPO_size)