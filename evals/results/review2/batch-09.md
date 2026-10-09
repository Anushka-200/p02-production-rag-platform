### n-b-follow_up-001  [follow_up, tenant B]
**Q:** Can you still participate if you don't want to do the whole thing?
history: user: How long does someone need to commit to the philosophy club if they want to go through all the topics? / assistant: One year
**Gold answer:** Yes, each topic stands alone and you can attend as many sessions as you want
- GOLD 1: posthog/people/philosophy-club.md #topics | path: Philosophy club > Topics | (all support quotes are in the gold passages)
  > See below for the full roadmap, with a link to each pre-read. This is a 1 year commitment if you want to do the whole thing, but each question stands alone, and you can attend as many sessions as you want! 1. [What counts as knowledge?](https://plato.stanford.edu/entries/epistemology/#WhatKnow) 2. [Can we trust our own reasoning?](https://plato.stanford.edu/entries/skepticism/#KnowJustSkep) 3. [If behavior is shaped by causes, what does accountability mean?](https://plato.stanford.edu/entries/freewill/#DoWeHaveFreeWill) 4. [Is success about happiness, achievement, or meaning?](https://plato.stanford.edu/entries/well-being/#TheWelBei) 5. [Are habits more important than intentions?](https://plato.stanford.edu/entries/ethics-virtue/#Virt) 6. [Should decisions prioritize results even when met [...]

### n-b-follow_up-002  [follow_up, tenant B]
**Q:** What about finding the ones who just started this week?
history: user: How can I find out who has admin or owner privileges in my account? / assistant: You can check the current_organization_membership_level property to identify admins and owners in your accounts.
**Gold answer:** You can identify new users who joined your account within the current week.
- GOLD 1: posthog/cs-and-onboarding/new-hire-onboarding-exercise.md #cohorts | path: New hire onboarding exercise > Tactical questions > Cohorts | (all support quotes are in the gold passages)
  > 1. Who are all the users in your accounts? 2. Who are all the admins / owners in your accounts? (Hint: check the current_organization_membership_level property) 3. Who are the new users in your account this week? 4. Who are the power users in your account? (Power users can be across multiple products, or you can split it by product. Define a power user as you see fit!)

### n-b-follow_up-003  [follow_up, tenant B]
**Q:** How is that different from sending a huge batch of requests all at once?
history: user: What should happen if a customer tells us they want to do load testing? / assistant: The pipeline team should be notified right away so they can prepare for any necessary scaling.
**Gold answer:** Sending a massive batch all at once isn't really load testing—it's more like a denial of service attack. Load testing involves gradually ramping up volume over time.
- GOLD 1: posthog/growth/sales/historical-import.md #load-testing | path: Historical import > Load testing | (all support quotes are in the gold passages)
  > If a customer mentions load testing, get answers to the above and then alert the pipeline team asap, so that accommodations can be made, as this may require scaling up to handle properly. If a customer plans to send a large volume of single capture requests all at once, rather than ramp up to a peak over some time period, that is not a load test but more like a denial of service (DoS).

### n-b-follow_up-004  [follow_up, tenant B]
**Q:** How do different teams prefer to receive these handoffs?
history: user: What should you do when a customer problem requires help from the product team? / assistant: You hand off the issue to the team that owns the product and can fix it.
**Gold answer:** Different teams have different preferences - some like written explanations sent asynchronously, while others want you to join their regular team meetings.
- GOLD 1: posthog/forward-deployed-engineering/working-with-product-engineering.md #escalations | path: Working with product engineering > Escalations | (all support quotes are in the gold passages)
  > When a customer engagement is blocked by something only a product team can fix, we hand off to the owning team. How that handoff works varies: some teams prefer async write-ups, others prefer you join their standup. Finding the best channel for each team is part of the job, the same way it is with customers. When something works well, share it with the team so others can use it.

### n-b-follow_up-005  [follow_up, tenant B]
**Q:** What should they do when bug reports or requests come in for their feature?
history: user: Who is responsible for keeping a PostHog feature running smoothly? / assistant: An Engineering owner is responsible for maintaining each feature, keeping it working well, suggesting improvements, planning launches, and ensuring it's properl
**Gold answer:** They need to review and prioritize each bug or request that arrives, though they can decide not to work on something if they think other work is more important.
- GOLD 1: posthog/engineering/feature-ownership.md #feature-ownership | path: Feature ownership | (all support quotes are in the gold passages)
  > Each feature at PostHog has an Engineering owner. This owner is responsible for maintaining the feature (keep the lights on), championing any efforts to improve it (e.g. by bringing up improvements in sprint planning), [planning launches](/handbook/marketing/product-announcements) for new parts of it, and making sure it is well documented. For shared developer tooling and infrastructure that cuts across product teams (CI, local dev, builds, migrations, etc.), see the [Developer Experience](/handbook/engineering/developer-experience) page. When a bug or feature request comes in, we tag it with the relevant label (see labels below). The owner is responsible for then prioritizing any bug/request that comes in for each feature. This does not mean working on every bug/request, an owner can ma [...]

### n-b-follow_up-006  [follow_up, tenant B]
**Q:** Where can I use that code to shop?
history: user: How do I get a discount code when I redeem a merch credit? / assistant: You click the Redeem button on the reward card, confirm the redemption, and then you receive a discount code.
**Gold answer:** You can click 'Use in store' to open the merch store with your code already applied, and then shop for anything you want.
- GOLD 1: posthog/community/points.mdx #merch-credits | path: Points & rewards > Redeeming points > Merch credits | (all support quotes are in the gold passages)
  > When you redeem a merch credit (gift card): 1. Click "Redeem" on the reward card 2. Confirm the redemption 3. You receive a discount code 4. Click "Use in store" to open the [merch store](/merch) with your code pre-applied 5. Shop for anything you'd like! Merch codes are saved in your transaction history, so you can always find them again if needed. ---

### n-b-unanswerable-001  [unanswerable, tenant B]
**Q:** How many fields can I add help_text to before the API documentation stops updating automatically?
**Gold answer:** 
- OTHER (posthog/engineering/posthog-com/api-docs.md | Editing the API docs): Editing the API docs The [PostHog API docs](/docs/api) are generated using [drf-spectacular](https://drf-spectacular.readthedocs.io/en/latest/readme.html). It looks at the Django models and djangorestframework serializers. > **Note:** We don't automatically add new API endpoints to the sidebar. You 
- OTHER (posthog/wizard-and-docs/api-specifications.mdx | API specifications > How to update the OpenAPI spec): API specifications > How to update the OpenAPI spec Any of the automatically generated pages sources from the OpenAPI spec. To update the content of an automatically generated page, you need to update the OpenAPI spec by making changes to the [PostHog/posthog](https://github.com/PostHog/posthog) rep
- OTHER (posthog/wizard-and-docs/api-specifications.mdx | date the OpenAPI spec > Updating the endpoint parameters and responses): API specifications > How to update the OpenAPI spec > Updating the endpoint parameters and responses The endpoint request body parameters, query parameters, path parameters, response body, response headers, API key scopes, etc. are all defined in the Django serializers and viewsets in the PostHog re

### n-b-unanswerable-002  [unanswerable, tenant B]
**Q:** When a new team member joins from a country we haven't hired from before, how long do we have to add the new flag sticker before their profile goes live?
**Gold answer:** 
- OTHER (posthog/engineering/posthog-com/add-team-member.mdx | Adding a team member > Notes): Adding a team member > Notes - If the new team member lives in a country we haven't hired from before, we'll need to add a new flag sticker for their country. Ask the in the `#posthogdotcom` Slack channel to do this. - We'll use the team member's public photo by default - They have an option to ask 
- OTHER (posthog/engineering/posthog-com/add-team-member.mdx | Adding a team member > Add the team member's photo): Adding a team member > Add the team member's photo 1. Watch for alerts that the new team member has been created (in the `#alerts-deel` private Slack channel). It's always worth keeping an eye on new hire announcements in `#general` in case the webhook doesn't fire. 1. Verify their preferred name in
- OTHER (posthog/onboarding/new-hire-onboarding.md | New hire onboarding > In-person onboarding): New hire onboarding > In-person onboarding Ideally, this will happen in Week 2-3, with the company of a few colleagues (depending on where we do it and who's around). It will be 3-4 days covering (among others): - Demo practice session with the team. - The data we track on customers in PostHog and s

### n-b-unanswerable-003  [unanswerable, tenant B]
**Q:** How many customer testimonials do we need to collect each quarter before we can use them on our website?
**Gold answer:** 
- OTHER (posthog/marketing/testimonials.md | Testimonials and G2 > Testimonials): Testimonials and G2 > Testimonials We speak to our users regularly and are often fortunate enough that they say nice things about our product or our way of working. Other times users talk about us in public, such as on social media or on review platforms and forums. Not all of the feedback we receiv
- OTHER (posthog/marketing/testimonials.md | Testimonials and G2 > Social reviews on G2): Testimonials and G2 > Social reviews on G2 We collect [reviews from users on G2](https://www.g2.com/products/posthog/reviews), both to act as social proof and to collect feedback on our product. After a process of trialling incentives, messaging and processes throughout 2022 we have established that
- OTHER (posthog/company/merch-store.md | Merch store > Merch giveaways > Customers): Merch store > Merch giveaways > Customers > Limit the use to one use only (_not_ one use per customer), otherwise it's unlimited free stuff for them, unlimited high cost for us! For feedback or general rewards we typically give users $30, which is enough for a t-shirt. For code contributions we tend

### n-b-unanswerable-004  [unanswerable, tenant B]
**Q:** What is the exact time that the kick-off session starts on my SuperDay?
**Gold answer:** 
- OTHER (posthog/people/hiring-process/sales-cs-hiring.md |  PostHog > Customer Success hiring process > Customer Success SuperDay): Sales and Customer Success Hiring > Sales and Customer Success hiring at PostHog > Customer Success hiring process > Customer Success SuperDay ##### What the SuperDay looks like The SuperDay is a paid full day of work (1,000 USD). You'll receive some tasks at the start of your day and submit your wo
- OTHER (posthog/people/hiring-process/index.mdx | Hiring process > Hiring stage overview > 4. PostHog SuperDay): Hiring process > Hiring stage overview > 4. PostHog SuperDay - (One day before the SuperDay) For non-technical roles, invite the candidate to a kickoff meeting with the hiring manager at the start of the day and send the candidate the task - aim to send this before the kick-off session so if the can
- OTHER (posthog/people/hiring-process/design-hiring.md |  Design hiring at PostHog > Design hiring process > 3. Design SuperDay): Design Hiring > Design hiring at PostHog > Design hiring process > 3. Design SuperDay The final stage of our interview process is the PostHog [SuperDay](/handbook/people/hiring-process#posthog-superday). This is a paid full day of work with us, which we can flexibly arrange around your schedule. A D

### n-b-unanswerable-005  [unanswerable, tenant B]
**Q:** How many hours per week should I dedicate to sharing knowledge and mentoring other team members?
**Gold answer:** 
- OTHER (posthog/getting-started/meetings.md | Meetings > Weekly schedule): Meetings > Weekly schedule - **Monday** - [PostHog all-hands meeting](https://posthog.com/handbook/exec/all-hands-topics). Members of the team share company-wide updates about things like recruitment, product metrics and commercial performance - the notes and recording are shared weekly in the [#gen
- OTHER (posthog/cs-and-onboarding/new-hire-onboarding-team-leads.md | Onboarding a new CSM (team leads) > In-person week (usually week 2)): Onboarding a new CSM (team leads) > In-person week (usually week 2) [In-person onboarding](/handbook/people/onboarding#in-person-onboarding) covers the company-wide process. Below is how we run the CSM week over 3-4 days - there's a [4-day agenda template](https://posthog.slack.com/docs/TSS5W8YQZ/F0
- OTHER (posthog/engineering/visiting-customers.md | Visiting customers as an engineer > 4. Plan the agenda): Visiting customers as an engineer > 4. Plan the agenda Work with your point of contact to set the agenda. Aim to get dedicated time with several people who use the product. A one-hour session is usually enough to go deep and uncover useful insights. Also offer other formats, like a company-wide trai

### n-b-unanswerable-006  [unanswerable, tenant B]
**Q:** How much does it cost to purchase time with the onboarding team?
**Gold answer:** 
- OTHER (posthog/people/onboarding.md | Onboarding > In-person onboarding): Onboarding > In-person onboarding While there is no fixed budget for onboardings they should be relatively less expensive than a small team offsite, which is $2,000 per person. Some considerations to reduce the cost: - Avoid intercontinental travel or choose a location that limits it to the minimum 
- OTHER (posthog/onboarding/onboarding-tracking.md | Onboarding process and tracking): Onboarding process and tracking The onboarding team operates a high volume, high velocity sales pipeline with all pay-as-you-go (or YC) accounts that are forecasted to spend > $500 and are not otherwise engaged by Sales/CSM. As such, Onboarding is a linear flow moving from initial outreach to confir
- OTHER (posthog/growth/sales/team-leads.md | am lead responsibilities > Team-specific responsibilities > Onboarding): Team lead responsibilities > Team-specific responsibilities > Onboarding The Onboarding team operates at scale, supporting hundreds of customers whose MRR falls below the TAM/CSM threshold. As a result, the number of customers in the program, as well as the ARR represented, can fluctuate from month 