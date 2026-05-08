# Governance

This document describes the governance model for the **TorchSurv** project.

## Project Mission

TorchSurv provides lightweight, flexible tools for deep survival analysis within
the PyTorch ecosystem. The project aims to lower the barrier to implementing
survival models by offering loss functions and evaluation metrics with a pure
PyTorch backend, free from restrictive parametric assumptions.

## Roles

### Maintainers

Maintainers have write access to the repository and are responsible for:

- Reviewing and merging pull requests
- Triaging issues
- Making release decisions
- Ensuring code quality and CI health

Current maintainers:

| Name | Affiliation | Role |
|------|-------------|------|
| [Thibaud Coroller](https://github.com/tcoroller) | Novartis | Creator, Maintainer |
| [Mélodie Monod](https://github.com/melodiemonod) | University Paris Dauphine – PSL | Creator, Maintainer |
| [Peter Krusche](https://github.com/pkrusche) | Novartis | Author, Maintainer |
| [Qian Cao](https://github.com/qiancao) | FDA | Author, Maintainer |

### Emeritus Contributors

Emeritus contributors have made significant past contributions but are no longer
actively maintaining the project:

- David Ohlssen (Novartis)
- Berkman Sahiner (FDA)
- Nicholas Petrick (FDA)

### Contributors

Anyone who contributes code, documentation, bug reports, or other improvements
to TorchSurv is a contributor. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to
get started.

Notable contributors:

- Yao Chen (Novartis)
- Sonia Dembowska (Novartis)
- Gene Pennello (FDA)

## Decision-Making

TorchSurv uses a **consensus-seeking** decision-making model:

1. **Proposals** are made via GitHub issues or pull requests.
2. Maintainers discuss and seek consensus. Silence after a reasonable review
   period (typically 7 days) is interpreted as agreement.
3. If consensus cannot be reached, a **simple majority vote** among maintainers
   decides the outcome.
4. Any maintainer may call for a vote on a contentious issue.

For day-to-day code changes, a single maintainer approval is sufficient to merge
a pull request, provided CI passes and no other maintainer has raised objections.

## Becoming a Maintainer

New maintainers are nominated by existing maintainers based on:

- Sustained, high-quality contributions over a period of time
- Demonstrated understanding of the project's goals and codebase
- Active participation in reviews and issue discussions

A nominee becomes a maintainer when approved by a majority of current
maintainers.

## Stepping Down

Maintainers may step down at any time by notifying the other maintainers. They
will be moved to the Emeritus Contributors list. A maintainer who has been
inactive for 12 months may be moved to emeritus status by majority vote after
an attempt to contact them.

## Conflict Resolution

If a dispute cannot be resolved through the normal decision-making process:

1. The involved parties should attempt to resolve it through direct discussion.
2. If unresolved, any maintainer may escalate to a formal vote.
3. As a last resort, the project defers to the
   [Linux Foundation Code of Conduct](https://lfprojects.org/policies/code-of-conduct/)
   enforcement process.

## Organizational Context

TorchSurv was created through a research collaboration between **Novartis** and
the **U.S. Food and Drug Administration (FDA)**. The project is open source under
the MIT License. While maintainers may be employed by these organizations, the
project's technical governance is independent — decisions are made by the
maintainer group as described above.

## Code of Conduct

This project follows the
[Linux Foundation Code of Conduct](https://lfprojects.org/policies/code-of-conduct/).
See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for details.

## Changes to Governance

Changes to this governance document require approval by a majority of current
maintainers via a pull request.
