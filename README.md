# GANs
Repo to learn about GANs. Images and some material taken from [Aladdin Persson's amazing Youtube channel.](https://www.youtube.com/channel/UCkzW5JSFwvKRjXABI-UTAkQ)

## Notes
Some examples of tasks currently being done with GANs:
- Data Augmentation
- Medical Applications (since data doesn't come from an actual patient, no privacy issues)
- Semi-supervised Learning (SGAN)

## What are GANs?
A class of ML techniques consisting of two networks playing an [adversarial game](https://towardsdatascience.com/a-game-theoretical-approach-for-adversarial-machine-learning-7523914819d5) against each other.

### Generator VS Discriminator: 
Both start randomly initialized and then simultaneously trained. In the end the generator produces elements indistinguishable from the "real" reference onces and the discriminator is forced to guess.

## Loss function:
![alt text][Disc_loss]





[Disc_loss] : https://github.com/Rodrigo163/GANs/screenshots/disc_loss.png "Discriminator loss function."
