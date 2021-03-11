# Contributions to pygwinc

The pygwinc project welcomes your contributions.  Our policy is that
all contributions should be peer-reviewed.  To facilitate the review,
please do not commit your work to this repository directly.  Instead,
please fork the repository and create a [merge
request](https://git.ligo.org/gwinc/pygwinc/-/merge_requests/new)
against the main pygwinc master branch.

When submitting code for merge, please follow good coding practice.
Respect the existing coding style, which for `pygwinc` is standard
[PEP8](https://www.python.org/dev/peps/pep-0008/) (with some
exceptions).  Make individual commits as logically distinct and atomic
as possible, and provide a complete, descriptive log of the changes
(including a top summary line).  Review efficiency follows code
legibility.

`pygwinc` comes with a validation command that can compare budgets
from the current code against those produced from different versions
in git (by default it compares against the current HEAD).  The command
can be run with:
```shell
$ python3 -m gwinc.test
```
Use the '--plot' or '--report' options to produce visual comparisons
of the noise differences.  The comparison can be done against an
arbitrary commit using the '--git-ref' option.  Traces for referenced
commits are cached, which speeds up subsequent comparison runs
significantly.

Once you submit your merge request a special CI job will determine if
there are budgets differences between your code and the master branch.
If there are, explicit approval from reviewers will be required before
your changes can be merged (see "approving noise" below).


## For reviewers: approving noise curve changes

As discussed above, merge requests that generate noise changes will
cause a pipeline failure in the `review:noise_change_approval` CI job.
The job will generate a report comparing the new noise traces against
those from master, which can be found under the 'View exposed
artifacts' menu item in the pipeline report.  Once you have reviewed
the report and the code, and understand and accept the noise changes,
click the 'Approve' button in the MR.  Once sufficient approval has
been given, `review:noise_change_approval` job can be re-run, which
should now pick up that approval has been given and allow the pipeline
to succeed.  Once the pipeline succeeds the merge request can be
merged.  Click the 'Merge' button to finally merge the code.
