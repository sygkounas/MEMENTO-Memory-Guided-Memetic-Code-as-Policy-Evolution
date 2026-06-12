cat > README.md <<'EOF'
# MEMENTO: Memory-Guided Memetic Code-as-Policy Evolution

MEMENTO is a framework for evolving executable code-as-policy programs for robotic manipulation and embodied-control tasks.

## Demos

The GIF demos below are stored in `assets/` and play directly in this README.

<table>
  <tr>
    <td align="center" width="50%">
      <img src="./assets/physical_franka_teaser.gif" width="100%">
      <br><b>Physical Franka</b>
    </td>
    <td align="center" width="50%">
      <img src="./assets/robosuite_teaser.gif" width="100%">
      <br><b>Robosuite</b>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="./assets/thor_demo1.gif" width="100%">
      <br><b>AI2-THOR Demo 1</b>
    </td>
    <td align="center" width="50%">
      <img src="./assets/thor_demo2.gif" width="100%">
      <br><b>AI2-THOR Demo 2</b>
    </td>
  </tr>
</table>

Full videos are available in the `videos/` folder.

## Installation

Install the required packages with:

```bash
pip install -r requirements.txt
