# IMC Prosperity 3 write-up

## Writers:

### James Cole 

https://www.linkedin.com/in/jamescole05/

### Tomás 

https://www.linkedin.com/in/tom%C3%A1s-barbosa-b47440156/

## Result


- #131 overall
- #70 Manual
- #10 in the UK
- ~14k teams 

## Introduction

IMC Prosperity 3 was a global algorithmic trading competition held over 15 days with over 14,000 participants competing. There were 5 different rounds each lasting 3 days with new challenges arising each round. The first round included ‘Rainforest Resin’, ‘Kelp’ as well as ‘Squid Ink’. Whilst further into the competition, new things were added such as ‘Magnificent Macarons’ and ‘Volcanic Rock’(Along with its call options). The underlying currency for the archipelago were seashells and the aim was to make the most amount of profit out of all the teams. In addition to the algorithmic section of the competition there was also a manual trading round consisting of game theory and mathematical problems.

## Tutorial 

As for the tutorial round it was very simple - there were 2 commodities available to trade (Rainforest Resin and Kelp) to help teams get used to how the trading system on the archipelago worked and to have a somewhat head-start in developing an algorithm for these commodities. Our approach to this was very similar for both; Resin allowed for market making and market taking as the fair value was set to 10,000 seashells throughout all rounds with wide bid/ask spreads allowing us to undercut and make profit very efficiently. Kelp was not trading around a single price level, but we noticed its price movements could be somewhat predicted by the state of the orderbook, so we quickly decided to use VWAP to calculate our fair value for the commodity and continue to market make/take.

## Round 1

In round 1 a new commodity to trade called Squid Ink was added, as well as the already 2 existing ones from the previous round. Fortunately, there weren't any changes that needed to be made for these, so we could pivot our focus primarily towards continuing to develop tools for the competition as well as an algorithm for squid ink. 
Throughout these 3 days we tried numerous basic methods to trade squid ink as we couldn’t continue to market make our way through. In the end we came to the decision to trade it using a mean reversion approach to scalp some profits. This didn’t turn out to be massively profitable. As the round finished and results were released we found we had ranked 1,200th globally which shocked us, but after further investigation we discovered our algorithm had actually been run on only 100k timestamps instead of the full 1 million like everybody else. After this had been resolved by the admins we jumped another 600 places globally at the start of round 2.
For the manual round, we had to convert between some currencies a maximum of 5 times and make money from profitable conversion rates. This was a trivial problem, as there was a relatively small number of currencies to convert to, so we simply wrote a script to brute force the optimal conversions and used that.

## Round 2

This round was based on ETF trading, a new concept to me entirely but I was able to conduct a sufficient amount of research and put it into practice within the first 24 hours. We approached this round using statistical arbitrage with the ETF (Picnic Baskets) against its synthetic basket composing of:
Six Croissants
Three Jams
One Djembe
Although this worked very well for us we weren’t able to profitably replicate this for picnic basket 2 so ultimately decided to not trade it in our final submission. Once submission had been run we climbed higher in the global ranking to 138th with a total profit of 208k seashells.

For the manual round, we had to trade flippers with turtles. They would have a reserve price and trade with any bids above their reserve price (in a first-price auction). Their reserve prices were uniformly distributed at random between 160-200 and 250-320, where the value of the flippers was 320. We were allowed two bids, but the second bid would only trade if it were also above the average of the other player's second bid. If it were below, the probability would fall cubically as a function of how far away the bid was from the average. 

<img width="1191" alt="Screenshot 2025-04-25 at 22 39 10" src="https://github.com/user-attachments/assets/6c73a971-fff7-4913-b239-01ce7f4aa864" />

For the manual round we had to pick one or two containers out of 10. Each container had a multiplier of a number of inhabitants. The payoff would then be 10k * multiplier / (inhabitants + % of players picking that crate). As you can see from the way the payoff is calculated, the multiplier/inhabitant ratio is a good metric to estimate how attractive a crate will be to other users, the inhabitant number by itself is a good metric for how much the payoff will change when more users select it, so we used it as a marker for risk.

Tomas was handling the manual round and he was rather busy this week so he only managed to look at it 1 hour before submission and had to take a simplistic approach, 73x multiplier was the one with the highest multiplier/inhabitant ratio and 90x had the highest multiplier, we thought those would be overpicked by low effort players so went with 80x crate, it had a decent ratio and a very high number of inhabitants which signified low risk.Our model for the payoff of each crate told us that no crate was likely to pay above 50k ()which was the fee to pick 2 crates instead of 1) So we only picked one, this turned out to be a good choice.

This didn’t give us a great payoff. We were probably around average, but it was a good, low-risk solution. It turned out that everyone went for high multiplier crates, so we got outperformed by low inhabitant (high risk) crate takers, which we were fine with, as we knew it was a possibility, but we didn’t want to take the extra risk.


## Round 3

Round 3 brought a lot of surprises to some of us, as a new equity called Volcanic Rock had been added along-side its’ european style call options that we could trade too.
 
For a significant part of our team, options were something new, so we chose to use a relatively simple strategy. We mapped options to implied volatilities and traded volatility in a mean-reversion manner. Because of the spread, deep ITM calls seemed to have a lot of noise when calculating the volatility because the extrinsic value was so close to 0, a 1$ increase or decrease due to the spread would make a huge difference in the IV, so we only traded ATM calls. 

Closer to the end, we had a hint from the organisers that it might be useful to plot the vol smile. This indeed was a more appropriate way to calculate IV, but we had previously assumed that the vol smile would be negligible because most people were rather unfamiliar with options, and we thought the organisers wouldn’t add such unnecessary complexity. With this hint we plotted the vol smile and it showed that it was indeed very relevant (As you can see in the figure below). But because our strategy relied on short term IV mean reversion and we were trading the options with different strikes independently, the moneyness shouldn’t change significantly unless there was a huge move in a short period of time so our strategy should still make sense for the most part despite not optimal and we didn’t have much time to perfect it.

<img width="1012" alt="Screenshot 2025-04-25 at 22 37 54" src="https://github.com/user-attachments/assets/f66138a6-4669-4bb8-ab46-e187147f8957" />

As the round came towards an end we had built a relatively strong (so we thought) algorithm for these options and their underlying. Not only this but some of us were able to learn so much about options and how they can be traded in different ways which had been totally foreign to them pre-prosperity. As results were released we were disappointed to discover we had dropped to 208th place. It seemed like some people had been able to integrate the volatility smile strategy faster than us, but also lots of people gambled with a directional trade on these options (most were punished on the following round where they lost a significant amount of money for trading this way).

With some elementary linear algebra, one can see that if there was no game theoretic aspect to the game where we have to consider other players' bids, the optimal bids would be 200 and 285. Despite this, we modelled the punishment function below the average and I believe being 2 seashells below the average was close to as bad as being 10 seashells above. Because of the asymmetric punishment, this looked to us like the “guess ⅔ of the average” game, where everyone wants to be slightly above the average as a way to protect themselves from risk. For this reason we were a little conservative and placed our bids at 200 and 302, the average ended up being only 287. We learnt from this that most players in the competition were taking a very simplistic approach and we should expect them to continue to do so in the future. Additionally, it was also clear to us that we had to take some more risks as our conservative approaches weren’t paying off. 

## Round 4

Round 4 was an interesting round to say the least. A new market had been added called ‘Pristine Cuisine’ alongside a new asset called Magnificent Macarons. These had also been added with other external indicators we could use to model our algorithm such as the sunlight index, sugar price and import/export tariffs too. We thought straight away a basic arbitraging approach to this round would give us a strong foundation to work on but we soon realised it wasn’t. Either our orders weren’t getting filled (even when flipping the spread) or our pnl would be very erratic as well as being quite volatile. After 3 days of sleepless nights trying to model everything we could using sunlight and running algorithms to search for any correlations between the given sunlight index and sugar prices we did find a decent correlation with macaron rallies when the sunlight index was low. We tried implementing a strategy to take advantage of current sunlight index and even trying to predict the future index using the current slope (as you can see in the figure below the slope rarely changes), but the API for interacting with pristine cuisine was so weird and poorly documented that we decided not to submit this algorithm as it could easily go wrong (and for many teams it did).  As the round concluded, we submitted a similar algorithm as the previous round with some small tweaks to certain parameters and logic. We all expected this to drop our placement significantly but it turned out we actually climbed to 38th place! This came as a massive shock to us as we were now 3rd in the UK too.

<img width="1117" alt="Screenshot 2025-04-25 at 22 35 22" src="https://github.com/user-attachments/assets/e9b1f2dc-2fd7-4705-b9a1-4f14d3e9ecd2" />


For the manual portion of this round, it was very similar to round 2, except there were 20 crates and we could pick up to 3 (second with a cost of 50k and third cost 100k). We modelled that the second crate was now worth it so we picked 2 (because the % of players per crate would be lower due to more options). We followed a similar strategy to round 2, as we believed people would be greedy and gravitate to high risk crates since they paid off well in r2, we stayed in midrange attractiveness, low risk crates (70x and 80x) and we had a significantly above average payoff, confirming that our predictions were correct.


## Round 5

 For the manual round we had a newspaper that gave us news regarding certain assets and we were able to place trades on them with a total capital of 1MM. But we had to pay fees to pay these trades and the fees grew quadratically with the allocation size, so even if we had the direction right, it wasn’t optimal to bet maximum size.

There was a similar manual challenge last year so we looked at it to understand the expected magnitude of the moves. With that, we analyzed the news and predicted how much each asset would change and then made an algorithm (available on this github) calculate the optimal allocation as a function of the predicted moves. This worked out pretty well, we made a good amount of seashells in this manual round. 

This round no new equities had been added but the bots that had been trading in the market amongst or with us had been de-anonymised and we were now able to look for patterns to trade with. Although during this round we all lacked a lot of time due to travelling, exams and other external factors we tried to put together an algo through mapping out all trades made between the bots. After 48 hours of near to no sleep we modelled an algorithm to trade more of the volcanic rock call options, squid ink, picnic basket 1 and the bots. We discovered Olivia always traded the very highs and lows of the market as well as Pablo profitably trading on Volcanic Rock which we were able to replicate. We tested this algorithm over all 3 million timestamps of provided data and the official IMC dashboard with extremely promising and safe results that were consistent throughout. We submitted this algorithm expecting to place in the top 25 globally, but this quickly proved to remain a dream. As we waited 72 hours for final results to be released we discovered our algorithm had actually lost a significant amount of money dropping us to 131st globally and 10th in the UK. We were extremely disappointed with our results but with the short amount of time we had to put into it as well as rigorous backtesting, we accepted our small defeat and learnt from our mistakes.

![image](https://github.com/user-attachments/assets/f552b4b2-26f6-4f0c-a0cc-03ba11dc96d7)

<img width="1086" alt="Screenshot 2025-04-25 at 22 40 24" src="https://github.com/user-attachments/assets/2c789128-11e2-49fe-aa44-fde8a01c0110" />

All in all we thoroughly enjoyed IMC Prosperity as it allowed us to develop python skills, mathematical skills, quantitative research and trading knowledge. We also met many new people along the way which all performed very well themselves. Although we were bitterly disappointed with our final placement we all had an amazing experience and learnt many new things, we also hope to place in the top 25 in Prosperity 4 next year!

Many thanks to IMC Trading and the firm responsible for running this competition.

