import { Component, Input, Output, EventEmitter } from '@angular/core';
import { Student } from '../student-input/student-input.component';

@Component({
  selector: 'app-student-item',
  standalone: true,
  imports: [],
  templateUrl: './student-item.component.html',
  styleUrl: './student-item.component.css'
})
export class StudentItemComponent {
  // Receives a single student object from the parent
  @Input() student!: Student;

  // Receives the display index (1-based) from the parent
  @Input() index!: number;

  // EventEmitter to notify parent when delete is clicked
  @Output() studentDeleted = new EventEmitter<number>();

  /**
   * Emits the student's id to the parent for deletion.
   */
  onDelete(): void {
    this.studentDeleted.emit(this.student.id);
  }
}
