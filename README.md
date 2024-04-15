# Title of the project

Documentation is **the most important** part of any scientific project. Please read the instructions here carefully and follow them. This will ensure:
1. You make good progress on your project in an organized manner
2. When you look back at your work one year from now, you will be able to still make sense of it and use it (it is surprisingly easy and common to become completely blank about old code, old decisions or old work in a long project!)
3. When you leave and someone else takes up your project, they are able to actually use the progress you made and move the project forward. This will benefit you as well when the project leads to a manuscript!

## How to use this repository for project management
You can read [this](https://rabernat.medium.com/scientific-collaboration-and-project-management-in-github-d74f2255ae5f) as well.

## 0. Basic instructions
1. Create a repository for your project using this repository as a template (see the `Use this remplate` button on the top right? Create a new repository from there!).
2. **Select `include all branches`** on the next page.
3. The owner should be `csndl-iitd`.
4. Give the repository a name (and description, if you want) based on your project.
5. Keep it public and hit `Create repository`.
6. Install `Github desktop` on your computer (not necessary but it will make your life easier).
7. Clone the newly created repository to your computer using Github desktop.
8. Now read the rest of this page to understand how to use your repository.

## 1. Project related discussion and weekly updates
All project related discussion will happen through `issues` on `Github.com`. Create the first issue named `Main discussion thread` if it is not already present and add the overall goal of the project there.  

In our weekly meetings, we will decide action items that you need to work on. These should be added appropriately as new issues or continued discussions in existing issues, and appropriately referenced in the Main discussion thread. Also assign the issues to yourself or whoever is responsible for them systematically.

For each weekly meeting you should create a `milestone` on Github.com. All the action items that we discuss for the next week should be assigned to the next milestone. Once you finish that action item, you can add the results to the corresponding issue. That way the results are automatically collected together in the milestone before our next meeting.

## 2. References
All project related reading you do should be added as PDFs to the references folder. The files should be named following the convention `author et al_year_title.pdf`. Refer to this file in your issues when you refer to the paper. Typically you will also have an issue related to reading the paper where you can summarize whatever was relevant in the paper to our project.  

**Remember to switch to the main branch before adding references to this folder.** Otherwise it will become difficult to track references scattered across branches. This matters only if you every create other branches for your analysis code.  

I recommend using `Zotero` for managing your references. It is easy to use, light-weight, free of charge and free of crap. And has all the goodies to integrate it with your choice word processors.

## 3. Experiments
If you are designing an experiment, all experiment files should go in the `experiment` folder. Please follow the following steps:
1. Create a new branch on Github named `experiment` if it does not already exist. Typically it should exist.
2. Install `github desktop` on the computer on which the experiment will live, and clone this repository there.
3. Switch to the `experiment` branch.
4. Create the experiment directly in the experiment folder of this repository only.
5. **Every time you update the experiment, commit the changes to git including a short description of what you changed.** You can use github desktop for doing this efficiently.

## 4. Analysis code

The `analysis` folder should be used for keeping all your code and Jupyter notebooks.  
1. **This folder must contain a `requirements.txt` file.** This file keeps track of all the Python packages and their versions that you use in your analysis.
2. Keep this file updated by regularly running `pip freeze > requirements.txt`
3. You can create a `scratch` branch or other branches for working if you want or need.
4. Add proper `library` and `notebooks` folders to keep the code organized nicely.
5. Commit regularly, and **surely when you complete an issue**, and include a proper commit message and description.

## 5. Data
All data should live in the `data` folder. There are two sub-folders by default - `raw` and `processed`. As the names suggest, raw data should go in `raw` folder. All the results of your processing should go in the `processed` folder. As needed, add other required structure to keep the data well-organized.  

As we use this system more, we will figure out how to handle large datasets. The readme will be updated accordingly.
