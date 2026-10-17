package infrastructure

import (
	"bytes"
	"cashguard/internal/domain"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"os"
	"os/exec"
	"strconv"
	"time"
)

// CommandAnalyzer runs a local executable. Video never leaves this machine.
// Arguments: executable <absolute-video-path> <budget-seconds>. stdout: completed result JSON, stderr: logs.
// The budget is what remains until the context deadline minus a reserve for frame encoding and saving.
type CommandAnalyzer struct{ Command string }

func (a CommandAnalyzer) Analyze(ctx context.Context, j domain.Job) (domain.Job, error) {
	if a.Command == "" {
		return j, fmt.Errorf("локальная модель не подключена: задайте ANALYZER_COMMAND")
	}
	budget := 270.0
	if deadline, ok := ctx.Deadline(); ok {
		budget = time.Until(deadline).Seconds() - 15
	}
	cmd := exec.CommandContext(ctx, a.Command, j.Path, strconv.FormatFloat(budget, 'f', 0, 64))
	var output limitedBuffer
	cmd.Stdout = &output
	cmd.Stderr = os.Stderr
	if e := cmd.Run(); e != nil {
		return j, fmt.Errorf("локальный анализатор завершился с ошибкой")
	}
	var result domain.Job
	if e := json.Unmarshal(output.Bytes(), &result); e != nil {
		return j, fmt.Errorf("анализатор вернул некорректный JSON")
	}
	if result.Status != "done" || (result.Verdict != "violation" && result.Verdict != "no_violation") {
		return j, fmt.Errorf("некорректный статус или вердикт анализатора")
	}
	if result.Verdict == "violation" && len(result.Violations) == 0 {
		return j, fmt.Errorf("нет доказательств нарушения")
	}
	if result.Verdict == "no_violation" && len(result.Violations) > 0 {
		return j, fmt.Errorf("вердикт противоречит списку нарушений")
	}
	for _, v := range result.Violations {
		if v.Type == "" || v.Description == "" || v.Timecode < 0 || len(v.Frames) == 0 {
			return j, fmt.Errorf("неполные данные нарушения")
		}
		for _, f := range v.Frames {
			if f.Timestamp < 0 {
				return j, fmt.Errorf("некорректный таймкод")
			}
			b, e := base64.StdEncoding.DecodeString(f.Image)
			if e != nil {
				return j, fmt.Errorf("некорректный кадр")
			}
			if _, _, e = image.DecodeConfig(bytes.NewReader(b)); e != nil {
				return j, fmt.Errorf("кадр не является изображением")
			}
		}
	}
	j.Verdict = result.Verdict
	j.Violations = result.Violations
	if j.Violations == nil {
		j.Violations = []domain.Violation{}
	}
	return j, nil
}

type limitedBuffer struct{ bytes.Buffer }

func (b *limitedBuffer) Write(p []byte) (int, error) {
	if b.Len()+len(p) > 32<<20 {
		return 0, fmt.Errorf("result too large")
	}
	return b.Buffer.Write(p)
}

type DemoAnalyzer struct{}

func (DemoAnalyzer) Analyze(ctx context.Context, j domain.Job) (domain.Job, error) {
	select {
	case <-ctx.Done():
		return j, ctx.Err()
	case <-time.After(3 * time.Second):
	}
	img := image.NewRGBA(image.Rect(0, 0, 640, 360))
	for y := 0; y < 360; y++ {
		for x := 0; x < 640; x++ {
			c := color.RGBA{28, 48, 57, 255}
			if x > 100 && x < 540 && y > 180 && y < 290 {
				c = color.RGBA{66, 101, 108, 255}
			}
			img.Set(x, y, c)
		}
	}
	var b bytes.Buffer
	_ = png.Encode(&b, img)
	j.Demo = true
	j.Verdict = "violation"
	j.Violations = []domain.Violation{{Type: "Демонстрационное событие", Timecode: 0, Description: "Тестовая карточка результата. Содержимое загруженного видео не анализировалось. Изображение — синтетическая иллюстрация, а не кадр видеозаписи.", Frames: []domain.Frame{{Timestamp: 0, Image: base64.StdEncoding.EncodeToString(b.Bytes())}}}}
	return j, nil
}
