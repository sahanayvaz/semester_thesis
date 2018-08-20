# semester_thesis
This repository contains the implementation of Proximal Policy Optimization[1] and Generative Adversarial Imitation Learning [2] to build a humanoid agent in OpenAI Gym to learn walking similar to humans.

The thesis report showing videos are located in the repo under semester_thesis(updated).pdf or via the link https://drive.google.com/file/d/1eAxMTBbJQHTjZ4Jba2QCxZLwn9jR1FfW/view?usp=sharing.

The work is based on the paper [3] with a difference of feeding separate state information to the generator and the discriminator. There is a slight jerkiness in walking behavior, and the reasons for it are still under investigation. I am trying to understand whether the jerkiness is caused by the instability of the modified humanoid model causing internal instabilities in MuJoCo physics engine, or I need to do an extensive hyperparameter search for the models. 

[1] SCHULMAN, J., WOLSKI, F., DHARIWAL, P., RADFORD, A., AND KLIMOV, O. 2017. Prox- imal policy optimization algorithms. CoRR abs/1707.06347.

[2] HO, J., AND ERMON, S. 2016. Generative adversarial imitation learning. In NIPS.

[3] MEREL, J., TASSA, Y., DHRUVA, T., SRINIVASAN, S., LEMMON, J., WANG, Z., WAYNE, G., AND HEESS, N. 2017. Learning human behaviors from motion capture by adversarial imitation. CoRR abs/1707.02201.
